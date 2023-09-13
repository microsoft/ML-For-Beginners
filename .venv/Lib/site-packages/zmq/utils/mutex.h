/*
* simplified from mutex.c from Foundation Library, in the Public Domain
* https://github.com/rampantpixels/foundation_lib/blob/master/foundation/mutex.c
*
* This file is Copyright (C) PyZMQ Developers
* Distributed under the terms of the Modified BSD License.
*
*/

#pragma once

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <pthread.h>
#endif

typedef struct {
#if defined(_WIN32)
    CRITICAL_SECTION csection;
#else
    pthread_mutex_t  mutex;
#endif
} mutex_t;


static void
_mutex_initialize(mutex_t* mutex) {
#if defined(_WIN32)
    InitializeCriticalSectionAndSpinCount(&mutex->csection, 4000);
#else
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex->mutex, &attr);
    pthread_mutexattr_destroy(&attr);
#endif
}

static void
_mutex_finalize(mutex_t* mutex) {
#if defined(_WIN32)
    DeleteCriticalSection(&mutex->csection);
#else
    pthread_mutex_destroy(&mutex->mutex);
#endif
}

mutex_t*
mutex_allocate(void) {
    mutex_t* mutex = (mutex_t*)malloc(sizeof(mutex_t));
    _mutex_initialize(mutex);
    return mutex;
}

void
mutex_deallocate(mutex_t* mutex) {
    if (!mutex)
        return;
    _mutex_finalize(mutex);
    free(mutex);
}

int
mutex_lock(mutex_t* mutex) {
#if defined(_WIN32)
    EnterCriticalSection(&mutex->csection);
    return 0;
#else
    return pthread_mutex_lock(&mutex->mutex);
#endif
}

int
mutex_unlock(mutex_t* mutex) {
#if defined(_WIN32)
    LeaveCriticalSection(&mutex->csection);
    return 0;
#else
    return pthread_mutex_unlock(&mutex->mutex);
#endif
}
