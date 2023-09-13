//-----------------------------------------------------------------------------
//  Copyright (c) 2010 Brian Granger, Min Ragan-Kelley
//
//  Distributed under the terms of the New BSD License.  The full license is in
//  the file LICENSE.BSD, distributed as part of this software.
//-----------------------------------------------------------------------------

#pragma once

#if defined(_MSC_VER)
#define pyzmq_int64_t __int64
#define pyzmq_uint32_t unsigned __int32
#else
#include <stdint.h>
#define pyzmq_int64_t int64_t
#define pyzmq_uint32_t uint32_t
#endif


#include "zmq.h"

#define _missing (-1)

#if (ZMQ_VERSION >= 40303)
    // libzmq >= 4.3.3 defines zmq_fd_t for us
    #define ZMQ_FD_T zmq_fd_t
#else
    #ifdef _WIN32
        #if defined(_MSC_VER) && _MSC_VER <= 1400
            #define ZMQ_FD_T UINT_PTR
        #else
            #define ZMQ_FD_T SOCKET
        #endif
    #else
        #define ZMQ_FD_T int
    #endif
#endif

#if (ZMQ_VERSION >= 40200)
    // Nothing to remove
#else
    #define zmq_curve_public(z85_public_key, z85_secret_key) _missing
#endif

// use unambiguous aliases for zmq_send/recv functions

#if ZMQ_VERSION_MAJOR >= 4
// nothing to remove
    #if ZMQ_VERSION_MAJOR == 4 && ZMQ_VERSION_MINOR == 0
        // zmq 4.1 deprecates zmq_utils.h
        // we only get zmq_curve_keypair from it
        #include "zmq_utils.h"
    #endif
#else
    #define zmq_curve_keypair(z85_public_key, z85_secret_key) _missing
#endif

// libzmq 4.2 draft API
#ifdef ZMQ_BUILD_DRAFT_API
    #if ZMQ_VERSION >= 40200
        #define PYZMQ_DRAFT_42
    #endif
#endif
#ifndef PYZMQ_DRAFT_42
    #define zmq_join(s, group) _missing
    #define zmq_leave(s, group) _missing
    #define zmq_msg_set_routing_id(msg, routing_id) _missing
    #define zmq_msg_routing_id(msg) 0
    #define zmq_msg_set_group(msg, group) _missing
    #define zmq_msg_group(msg) NULL
#endif

#if ZMQ_VERSION >= 40100
// nothing to remove
#else
    #define zmq_msg_gets(msg, prop) _missing
    #define zmq_has(capability) _missing
    #define zmq_proxy_steerable(in, out, mon, ctrl) _missing
#endif

#if ZMQ_VERSION_MAJOR >= 3
    #define zmq_sendbuf zmq_send
    #define zmq_recvbuf zmq_recv

    // 3.x deprecations - these symbols haven't been removed,
    // but let's protect against their planned removal
    #define zmq_device(device_type, isocket, osocket) _missing
    #define zmq_init(io_threads) ((void*)NULL)
    #define zmq_term zmq_ctx_destroy
#else
    #define zmq_ctx_set(ctx, opt, val) _missing
    #define zmq_ctx_get(ctx, opt) _missing
    #define zmq_ctx_destroy zmq_term
    #define zmq_ctx_new() ((void*)NULL)

    #define zmq_proxy(a,b,c) _missing

    #define zmq_disconnect(s, addr) _missing
    #define zmq_unbind(s, addr) _missing

    #define zmq_msg_more(msg) _missing
    #define zmq_msg_get(msg, opt) _missing
    #define zmq_msg_set(msg, opt, val) _missing
    #define zmq_msg_send(msg, s, flags) zmq_send(s, msg, flags)
    #define zmq_msg_recv(msg, s, flags) zmq_recv(s, msg, flags)

    #define zmq_sendbuf(s, buf, len, flags) _missing
    #define zmq_recvbuf(s, buf, len, flags) _missing

    #define zmq_socket_monitor(s, addr, flags) _missing

#endif
