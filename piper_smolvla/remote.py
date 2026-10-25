"""ssh and rsync to a rented GPU box. Plain OpenSSH and rsync: host aliases, keys and jump hosts come from
~/.ssh/config, so `--remote gpu` works for anything `ssh gpu` reaches."""
from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field


@dataclass
class Remote:
    host: str                       # user@host or an ~/.ssh/config alias
    port: int | None = None
    ssh_opts: list[str] = field(default_factory=lambda: ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                                                         "-o", "ServerAliveInterval=30"])
    dry: bool = False               # print the commands instead of running them
    log: list = field(default_factory=list)

    def _ssh_base(self) -> list[str]:
        base = ["ssh", *self.ssh_opts]
        if self.port:
            base += ["-p", str(self.port)]
        return base

    def ssh(self, command: str, check: bool = True, capture: bool = True, timeout: float | None = 120) -> str:
        """Run one shell command on the box. The command string is the remote shell's to parse; quote values
        with shlex.quote."""
        argv = self._ssh_base() + [self.host, command]
        self.log.append(argv)
        if self.dry:
            print("  $", shlex.join(argv))
            return ""
        r = subprocess.run(argv, capture_output=capture, text=True, timeout=timeout)
        if check and r.returncode != 0:
            raise RuntimeError(f"ssh {self.host} failed ({r.returncode}): {(r.stderr or r.stdout or '').strip()[-800:]}")
        return (r.stdout or "") if capture else ""

    def rsync(self, src: str, dst: str, excludes: tuple[str, ...] = (), delete: bool = False, copy_links: bool = False,
              timeout: float | None = None) -> None:
        """src/dst use rsync syntax; a remote side is written ':path' and gets the host prefixed here."""
        e = shlex.join(self._ssh_base())
        argv = ["rsync", "-az", "--partial", "-e", e]
        if copy_links:
            argv.append("-L")
        if delete:
            argv.append("--delete")
        for x in excludes:
            argv += ["--exclude", x]
        fix = lambda s: f"{self.host}{s}" if s.startswith(":") else s
        argv += [fix(src), fix(dst)]
        self.log.append(argv)
        if self.dry:
            print("  $", shlex.join(argv))
            return
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if r.returncode not in (0, 24):  # 24: files vanished while copying (a checkpoint being rotated)
            raise RuntimeError(f"rsync {src} -> {dst} failed ({r.returncode}): {r.stderr.strip()[-800:]}")
