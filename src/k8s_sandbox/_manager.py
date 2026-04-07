from __future__ import annotations

import asyncio
from contextvars import ContextVar

from rich import box, print
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from k8s_sandbox._helm import Release, _delete_namespace, _namespace_per_sample_enabled, get_all_releases
from k8s_sandbox._helm import uninstall as helm_uninstall
from k8s_sandbox._kubernetes_api import get_current_context_name, get_default_namespace


class HelmReleaseManager:
    """
    Manages the lifecycle of Helm releases.

    Each instance of this class is scoped to a single async context.
    """

    _context_var: ContextVar[HelmReleaseManager] = ContextVar("k8s_manager_instance")

    def __init__(self) -> None:
        self._installed_releases: list[Release] = []

    @classmethod
    def get_instance(cls) -> HelmReleaseManager:
        """Gets the Manager instance for the current async context."""
        try:
            return cls._context_var.get()
        except LookupError:
            manager = cls()
            cls._context_var.set(manager)
            return manager

    async def install(self, release: Release) -> None:
        """
        Installs a release and tracks it for eventual cleanup.

        Args:
          release (Release): The release to install and track.
        """
        # Track the release regardless of the install result.
        self._installed_releases.append(release)
        await release.install()

    async def uninstall(self, release: Release, quiet: bool) -> None:
        """
        Uninstalls a release managed by this instance.

        Args:
          release (Release): The release to uninstall.
          quiet (bool): If True, suppress output to the console.
        """
        await release.uninstall(quiet)
        self._installed_releases.remove(release)

    async def uninstall_all(self, print_only: bool) -> None:
        """Uninstalls all releases managed by this instance.

        This method is not quiet i.e. it will print output to the console.

        Args:
          print_only (bool): If True, print cleanup instructions without actually
            uninstalling anything.
        """
        if len(self._installed_releases) == 0:
            return
        if print_only:
            self._print_cleanup_instructions()
            return
        _print_do_not_interrupt()
        tasks = [release.uninstall(quiet=False) for release in self._installed_releases]
        # Clear the list before awaiting the tasks to prevent other calls to this method
        # from interfering.
        self._installed_releases.clear()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _print_cleanup_instructions(self) -> None:
        table = Table(
            title="K8s Sandbox Releases (not yet cleaned up):",
            box=box.SQUARE_DOUBLE_HEAD,
            show_lines=True,
            title_style="bold",
            title_justify="left",
        )
        table.add_column("Release(s)", no_wrap=True)
        table.add_column("Namespace")
        table.add_column("Cleanup")
        for release in self._installed_releases:
            table.add_row(
                release.release_name,
                release._namespace,
                f"[blue]inspect sandbox cleanup k8s {release.release_name}[/blue]",
            )
        print("")
        print(table)
        print(
            "\nCleanup all sandbox releases with: "
            "[blue]inspect sandbox cleanup k8s[/blue]\n"
        )


async def uninstall_unmanaged_release(release_name: str) -> None:
    """
    Uninstall a Helm release which is not managed by a HelmReleaseManager.

    Searches across all namespaces to find the release.

    Args:
      release_name (str): The name of the release to uninstall (e.g. "lsphdyup").
    """
    _print_do_not_interrupt()
    # Search all namespaces to find the release.
    all_releases = await get_all_releases(context_name=None)
    match = next((ns for name, ns in all_releases if name == release_name), None)
    if match is None:
        # Fall back to default namespace for backwards compatibility.
        match = get_default_namespace(context_name=None)
    await helm_uninstall(release_name, match, context_name=None, quiet=False)
    if _namespace_per_sample_enabled():
        await _delete_namespace(match, context_name=None)


async def uninstall_all_unmanaged_releases() -> None:
    def _print_table(releases: list[tuple[str, str]]) -> None:
        print("Releases to be uninstalled:")
        table = Table(
            box=box.SQUARE,
            show_lines=False,
            title_style="bold",
            title_justify="left",
        )
        table.add_column("Release")
        table.add_column("Namespace")
        for release, namespace in releases:
            table.add_row(f"[red]{release}[/red]", namespace)
        print(table)

    releases = await get_all_releases(context_name=None)
    if len(releases) == 0:
        print(
            f"No Inspect sandbox releases found in your "
            f"current Kubernetes context '{get_current_context_name()}'."
        )
        return
    _print_table(releases)
    namespaces = sorted({ns for _, ns in releases})
    ns_label = ", ".join(f"'{ns}'" for ns in namespaces)
    if not Confirm.ask(
        f"Are you sure you want to uninstall ALL {len(releases)} Inspect sandbox "
        f"release(s) in {ns_label}? If this is a shared namespace, "
        "this may affect other users.",
    ):
        print("Cancelled.")
        return
    tasks = [
        helm_uninstall(release, namespace, context_name=None, quiet=False)
        for release, namespace in releases
    ]
    await asyncio.gather(*tasks, return_exceptions=True)
    # Clean up per-sample namespaces.
    default_ns = get_default_namespace(context_name=None)
    per_sample_namespaces = [ns for ns in namespaces if ns != default_ns]
    if per_sample_namespaces:
        ns_tasks = [
            _delete_namespace(ns, context_name=None) for ns in per_sample_namespaces
        ]
        await asyncio.gather(*ns_tasks, return_exceptions=True)
    print("Complete.")


def _print_do_not_interrupt() -> None:
    print(
        Panel(
            "[bold][blue]Cleaning up K8s resources (please do not interrupt this "
            "operation!):[/blue][/bold]",
        )
    )
