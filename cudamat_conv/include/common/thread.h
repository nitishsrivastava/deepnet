/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef THREAD_H_
#define THREAD_H_
#include <pthread.h>
#include <stdio.h>
#include <errno.h>
#include <assert.h>

/*
 * Abstract joinable thread class.
 * The only thing the implementer has to fill in is the run method.
 */
class Thread {
private:
    pthread_attr_t _pthread_attr;
    pthread_t _threadID;
    bool _joinable, _startable;

    static void* start_pthread_func(void *obj) {
        void* retval = reinterpret_cast<Thread*>(obj)->run();
        pthread_exit(retval);
        return retval;
    }
protected:
    virtual void* run() = 0;
public:
    Thread(bool joinable) : _joinable(joinable), _startable(true) {
        pthread_attr_init(&_pthread_attr);
        pthread_attr_setdetachstate(&_pthread_attr, joinable ? PTHREAD_CREATE_JOINABLE : PTHREAD_CREATE_DETACHED);
    }

    virtual ~Thread() {
    }

    pthread_t start() {
        assert(_startable);
        _startable = false;
        int n;
        if ((n = pthread_create(&_threadID, &_pthread_attr, &Thread::start_pthread_func, (void*)this))) {
            errno = n;
            perror("pthread_create error");
        }
        return _threadID;
    }

    void join(void **status) {
        assert(_joinable);
        int n;
        if((n = pthread_join(_threadID, status))) {
            errno = n;
            perror("pthread_join error");
        }
    }

    void join() {
        join(NULL);
    }

    pthread_t getThreadID() const {
        return _threadID;
    }
};

#endif /* THREAD_H_ */
