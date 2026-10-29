#pragma once

#include <stddef.h>
#include <sys/types.h>

struct netio_desc;
typedef int (*netio_udp_handler)(struct netio_desc *, unsigned char *, ssize_t,
                                 void *, void *);

#ifdef __cplusplus
extern "C" {
#endif
int netio_udp_rxl_init(void);
void netio_udp_rxl_shutdown(void);
int netio_udp_rxl_add(struct netio_desc *, netio_udp_handler, void *, void *);
int netio_udp_rxl_remove(struct netio_desc *, int all);
int netio_udp_rxl_connect(struct netio_desc *, char *, int);
ssize_t netio_receive_dispatch(struct netio_desc *, void *, ssize_t);
#ifdef __cplusplus
}
#endif
