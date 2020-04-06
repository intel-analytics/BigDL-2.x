# import re
# from horovod.run.util import network
#
# def get_common_intfs(all_host_names, verbose):
#
#     if verbose >= 2:
#         print('Filtering local host names.')
#     remote_host_names = network.filter_local_addresses(all_host_names)
#     if verbose >= 2:
#         print('Remote host found: ' + ' '.join(remote_host_names))
#
#     if len(remote_host_names) > 0:
#         if verbose >= 2:
#             print('Testing interfaces on all the hosts.')
#
#         local_host_names = set(all_host_names) - set(remote_host_names)
#         # Find the set of common, routed interfaces on all the hosts (remote
#         # and local) and specify it in the args to be used by NCCL. It is
#         # expected that the following function will find at least one interface
#         # otherwise, it will raise an exception.
#         common_intfs = _driver_fn(all_host_names, local_host_names,
#                                   settings, fn_cache=fn_cache)
#
#         if verbose >= 2:
#             print('Interfaces on all the hosts were successfully checked.')
#             print('Common interface found: ' + ' '.join(common_intfs))
#
#     else:
#         if verbose >= 2:
#             print('All hosts are local, finding the interfaces '
#                   'with address 127.0.0.1')
#         # If all the given hosts are local, find the interfaces with address
#         # 127.0.0.1
#         common_intfs = set()
#         for iface, addrs in net_if_addrs().items():
#             if settings.nic and iface != settings.nic:
#                 continue
#             for addr in addrs:
#                 if addr.family == AF_INET and addr.address == '127.0.0.1':
#                     common_intfs.add(iface)
#                     break
#
#         if len(common_intfs) == 0:
#             raise ValueError('No interface is found for address 127.0.0.1.')
#
#         if verbose >= 2:
#             print('Local interface found ' + ' '.join(common_intfs))
#
#     # get the driver IPv4 address
#     driver_ip = _get_driver_ip(common_intfs)