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

#ifndef QUEUE_H_
#define QUEUE_H_
#include <pthread.h>
#include <stdlib.h>

/*
 * A thread-safe circular queue that automatically grows but never shrinks.
 */
template <class T>
class Queue {
private:
    T *_elements;
    int _numElements;
    int _head, _tail;
    int _maxSize;
    pthread_mutex_t *_queueMutex;
    pthread_cond_t *_queueCV;

    void _init(int initialSize) {
        _numElements = 0;
        _head = 0;
        _tail = 0;
        _maxSize = initialSize;
        _elements = new T[initialSize];
        _queueCV = (pthread_cond_t*)(malloc(sizeof (pthread_cond_t)));
        _queueMutex = (pthread_mutex_t*)(malloc(sizeof (pthread_mutex_t)));
        pthread_mutex_init(_queueMutex, NULL);
        pthread_cond_init(_queueCV, NULL);
    }

    void expand() {
        T *newStorage = new T[_maxSize * 2];
        memcpy(newStorage, _elements + _head, (_maxSize - _head) * sizeof(T));
        memcpy(newStorage + _maxSize - _head, _elements, _tail * sizeof(T));
        delete[] _elements;
        _elements = newStorage;
        _head = 0;
        _tail = _numElements;
        _maxSize *= 2;
    }
public:
    Queue(int initialSize) {
        _init(initialSize);
    }

    Queue()  {
        _init(1);
    }

    ~Queue() {
        pthread_mutex_destroy(_queueMutex);
        pthread_cond_destroy(_queueCV);
        delete[] _elements;
        free(_queueMutex);
        free(_queueCV);
    }

    void enqueue(T el) {
        pthread_mutex_lock(_queueMutex);
        if(_numElements == _maxSize) {
            expand();
        }
        _elements[_tail] = el;
        _tail = (_tail + 1) % _maxSize;
        _numElements++;

        pthread_cond_signal(_queueCV);
        pthread_mutex_unlock(_queueMutex);
    }

    /*
     * Blocks until not empty.
     */
    T dequeue() {
        pthread_mutex_lock(_queueMutex);
        if(_numElements == 0) {
            pthread_cond_wait(_queueCV, _queueMutex);
        }
        T el = _elements[_head];
        _head = (_head + 1) % _maxSize;
        _numElements--;
        pthread_mutex_unlock(_queueMutex);
        return el;
    }

    /*
     * Obviously this number can change by the time you actually look at it.
     */
    inline int getNumElements() const {
        return _numElements;
    }
};

#endif /* QUEUE_H_ */
