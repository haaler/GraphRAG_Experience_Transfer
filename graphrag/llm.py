"""Helpers for sending prompts to a language model.

There are two options to pick from:

- :class: OllamaLLM — talks to a local Ollama server.
- :class: OpenAILLM — talks to an OpenAI-style HTTP API (used for Azure).

Both have the same 'invoke(prompt, system_msg, num_ctx) -> str' method
and behave the same way when something goes wrong: 
every call runs on a worker thread so it can be cut off if it takes too long, 
and temporary errors (timeouts, rate limits) are retried a few times with a growing wait between attempts.
The shared work lives on :class: BaseLLM; each subclass only adds the actual API call 
and an extra check for errors it knows how to spot.
"""

import time
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import ollama


class BaseLLM:
    """Shared parts of the two LLM classes.

    Each subclass plugs in its own API call through a _call_* method.
    The retry loop and the chat-message list are built once here so the two subclasses don't have to repeat them.
    """

    # Subclasses set these in __init__.
    max_retries: int
    retry_delay: float
    timeout: float
    _executor: ThreadPoolExecutor

    def invoke(
        self,
        prompt: str,
        system_msg: Optional[str] = None,
        num_ctx: Optional[int] = None,
    ) -> str:
        raise NotImplementedError

    @staticmethod
    def _build_messages(prompt: str, system_msg: Optional[str]) -> list:
        """Build the chat-message list both subclasses send to their APIs."""
        return [
            {"role": "system",
             "content": system_msg or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

    def _classify_error(self, e: Exception, extra_check) -> tuple:
        """Decide whether an error is a timeout or a rate-limit so the retry loop can handle it.
        Returns '(is_timeout, is_rate_limit)'.
        """
        err = str(e)
        err_lower = err.lower()
        type_name = type(e).__name__

        is_timeout = (
            "timeout" in err_lower
            or "ReadTimeout" in type_name
            or "ConnectTimeout" in type_name
        )
        if extra_check is not None and extra_check(e):
            is_timeout = True

        is_rate_limit = "429" in err or "too many" in err_lower

        return is_timeout, is_rate_limit

    def _log_retry(self, attempt: int, reason: str, elapsed: float, wait: float):
        """Print a single-line retry notice."""
        print(
            f"  [retry {attempt + 1}/{self.max_retries}] "
            f"{reason} after {elapsed:.0f}s, retrying in {wait:.0f}s..."
        )

    def _invoke_with_retry(
        self,
        call_fn: Callable[[], str],
        *,
        label: str,
        extra_retryable_check: Optional[Callable[[Exception], bool]] = None,
    ) -> str:
        """Run 'call_fn' on a worker thread. Give up if it takes too long.
        Try again on temporary errors.

        Args:
            call_fn: A function that takes no arguments and runs the API call.
            label: Name of the model service, used in log messages ("Ollama" or "Azure").
            extra_retryable_check: Optional function that looks at an error and decides whether it should
                trigger a retry, used for error patterns specific to one of the two services
                (e.g. Ollama returning HTTP 504).
        """
        for attempt in range(self.max_retries):
            is_last = (attempt == self.max_retries - 1)
            t0 = time.time()

            try:
                future = self._executor.submit(call_fn)
                return future.result(timeout=self.timeout)

            except FuturesTimeoutError:
                elapsed = time.time() - t0
                future.cancel()

                if is_last:
                    raise TimeoutError(
                        f"LLM call timed out after {self.max_retries} attempts "
                        f"({self.timeout}s each)"
                    )

                wait = self.retry_delay * (attempt + 1)
                self._log_retry(attempt, "Wall-clock timeout", elapsed, wait)
                time.sleep(wait)

            except Exception as e:
                elapsed = time.time() - t0
                is_timeout, is_rate_limit = self._classify_error(e, extra_retryable_check)

                if is_last or not (is_timeout or is_rate_limit):
                    raise

                multiplier = 2 if is_rate_limit else 1
                wait = self.retry_delay * (attempt + 1) * multiplier
                reason = "Rate limited (429)" if is_rate_limit else f"{label} timeout"

                self._log_retry(attempt, reason, elapsed, wait)
                time.sleep(wait)


class OllamaLLM(BaseLLM):
    """Talks to a local Ollama server, with a time limit and retries on temporary errors."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        num_ctx: int = 4096,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        timeout: float = 120.0,
    ):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        self._client = ollama.Client(timeout=timeout)

        # 8 worker threads to speed up invoke() calls (entity extraction runs up to EXTRACTION_WORKERS=4 in parallel).
        self._executor = ThreadPoolExecutor(max_workers=8)

    def _call_ollama(self, messages, num_ctx):
        """The actual Ollama chat call, run inside a worker thread."""
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_ctx": num_ctx,
            },
        )

        return resp["message"]["content"].strip()

    def invoke(
        self,
        prompt: str,
        system_msg: Optional[str] = None,
        num_ctx: Optional[int] = None,
    ) -> str:
        """Send a prompt to Ollama"""
        ctx = num_ctx if num_ctx is not None else self.num_ctx
        messages = self._build_messages(prompt, system_msg)

        return self._invoke_with_retry(
            lambda: self._call_ollama(messages, ctx),
            label="Ollama",
            extra_retryable_check=lambda e: "504" in str(e),
        )


class OpenAILLM(BaseLLM):
    """Talks to an OpenAI-style HTTP API (used for Azure), with the same time limit and retries as OllamaLLM."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        timeout: float = 120.0,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

        self._executor = ThreadPoolExecutor(max_workers=8)

    def _call_openai(self, messages):
        """The actual OpenAI chat call"""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        return resp.choices[0].message.content.strip()

    def invoke(
        self,
        prompt: str,
        system_msg: Optional[str] = None,
        num_ctx: Optional[int] = None,
    ) -> str:
        """Send a prompt to the API (Azure OpenAI)"""
        messages = self._build_messages(prompt, system_msg)

        return self._invoke_with_retry(
            lambda: self._call_openai(messages),
            label="Azure",
            extra_retryable_check=lambda e: "rate" in str(e).lower(),
        )
