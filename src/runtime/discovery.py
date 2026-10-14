"""局域网运行模式服务发现广播。"""

from __future__ import annotations

import ipaddress
import logging
import socket

from zeroconf import ServiceInfo, Zeroconf

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_pipelineA._tcp.local."


def _is_usable_ipv4(ip: str) -> bool:
    """过滤掉回环、未指定和链路本地地址。"""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return (
        isinstance(addr, ipaddress.IPv4Address)
        and not addr.is_loopback
        and not addr.is_unspecified
        and not addr.is_link_local
    )


def _detect_ipv4_address() -> str:
    """尽量获取局域网可达的 IPv4 地址。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("224.0.0.251", 5353))
            ip = sock.getsockname()[0]
            if _is_usable_ipv4(ip):
                return ip
    except OSError:
        logger.debug("通过 UDP 探测本机 IP 失败", exc_info=True)

    hostnames = [socket.gethostname()]
    fqdn = socket.getfqdn()
    if fqdn not in hostnames:
        hostnames.append(fqdn)

    for name in hostnames:
        try:
            for family, _, _, _, sockaddr in socket.getaddrinfo(name, None, socket.AF_INET):
                if family != socket.AF_INET:
                    continue
                ip = sockaddr[0]
                if _is_usable_ipv4(ip):
                    return ip
        except OSError:
            logger.debug("解析主机名 %s 失败", name, exc_info=True)

    logger.warning("未找到可用局域网 IPv4 地址，回退为 127.0.0.1")
    return "127.0.0.1"


class RuntimeAnnouncer:
    """运行模式下，将 gRPC 服务通过 mDNS 广播到局域网。"""

    def __init__(
        self,
        port: int,
        service_name: str = "edge-srv",
        props: dict[str, str] | None = None,
    ) -> None:
        self._registered = False
        self._zc = Zeroconf()
        ip = _detect_ipv4_address()
        properties = {
            "version": "0.4",
            "mode": "runtime",
            **(props or {}),
        }
        self.info = ServiceInfo(
            type_=SERVICE_TYPE,
            name=f"{service_name}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip)],
            port=port,
            properties=properties,
            server=f"{service_name}.local.",
        )

    def start(self) -> None:
        """发布服务。"""
        if self._registered:
            return
        self._zc.register_service(self.info)
        self._registered = True
        logger.info("mDNS published at %s:%s", SERVICE_TYPE, self.info.port)

    def stop(self) -> None:
        """撤销服务广播。"""
        if self._registered:
            self._zc.unregister_service(self.info)
            self._registered = False
        self._zc.close()
