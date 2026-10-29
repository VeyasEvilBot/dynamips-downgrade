#pragma once

#include <stddef.h>
#include <sys/types.h>

struct virtual_tty;

struct vtty_io_event {
  struct virtual_tty *owner;
  int fd;
  int connected;
  const unsigned char *data;
  size_t size;
  void *handle;
};

#ifdef __cplusplus
extern "C" {
#endif
int vtty_io_start(void);
void vtty_io_stop(void);
void vtty_io_drain(void);
int vtty_io_attach(struct virtual_tty *, int, const int *, size_t);
int vtty_io_detach(struct virtual_tty *);
ssize_t vtty_io_write(struct virtual_tty *, const void *, size_t);
ssize_t vtty_io_write_fd(int, const void *, size_t);
int vtty_io_take(struct vtty_io_event *);
int vtty_io_active(const struct vtty_io_event *);
void vtty_io_complete(struct vtty_io_event *);
#ifdef __cplusplus
}
#endif
