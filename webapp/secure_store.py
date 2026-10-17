"""OS-backed secret storage for saved dashboard endpoint credentials."""
from __future__ import annotations


SERVICE_NAME = "TradingAgents Dashboard OpenAI-compatible endpoints"


class SecretStoreError(RuntimeError):
    """Raised when a secure OS credential store is unavailable or fails."""


class KeyringSecretStore:
    """Use a recommended keyring backend; never fall back to plaintext storage."""

    @staticmethod
    def _keyring():
        try:
            import keyring
        except ImportError as exc:
            raise SecretStoreError(
                "Secure credential storage is unavailable. Install the web extra with keyring support."
            ) from exc
        try:
            backend = keyring.get_keyring()
            if not keyring.core.recommended(backend):
                raise SecretStoreError(
                    "No secure operating-system keyring is available; refusing to store an API key."
                )
        except SecretStoreError:
            raise
        except Exception as exc:
            raise SecretStoreError(
                "Could not access the operating-system credential store. Unlock it and try again."
            ) from exc
        return keyring

    def set_secret(self, key: str, value: str) -> None:
        try:
            self._keyring().set_password(SERVICE_NAME, key, value)
        except SecretStoreError:
            raise
        except Exception as exc:
            raise SecretStoreError(
                "Could not save the API key to the operating-system credential store."
            ) from exc

    def get_secret(self, key: str) -> str | None:
        try:
            return self._keyring().get_password(SERVICE_NAME, key)
        except SecretStoreError:
            raise
        except Exception as exc:
            raise SecretStoreError(
                "Could not read the API key from the operating-system credential store."
            ) from exc

    def delete_secret(self, key: str) -> None:
        try:
            keyring = self._keyring()
            if keyring.get_password(SERVICE_NAME, key) is not None:
                keyring.delete_password(SERVICE_NAME, key)
        except SecretStoreError:
            raise
        except Exception as exc:
            raise SecretStoreError(
                "Could not remove the API key from the operating-system credential store."
            ) from exc
