/*

Platform-independant detection of IPC path max length

Copyright (c) 2012 Godefroid Chapelle

Distributed under the terms of the New BSD License.  The full license is in
the file LICENSE.BSD, distributed as part of this software.
 */

#pragma once

#if defined(HAVE_SYS_UN_H)
#if defined _MSC_VER
#include <afunix.h>
#else
#include <sys/un.h>
#endif
int get_ipc_path_max_len(void) {
    struct sockaddr_un *dummy;
    return sizeof(dummy->sun_path) - 1;
}
#else
int get_ipc_path_max_len(void) {
    return 0;
}
#endif
