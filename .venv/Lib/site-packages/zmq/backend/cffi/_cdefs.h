void zmq_version(int *major, int *minor, int *patch);

void* zmq_socket(void *context, int type);
int zmq_close(void *socket);

int zmq_bind(void *socket, const char *endpoint);
int zmq_connect(void *socket, const char *endpoint);

int zmq_errno(void);
const char * zmq_strerror(int errnum);

int zmq_device(int device, void *frontend, void *backend);

int zmq_unbind(void *socket, const char *endpoint);
int zmq_disconnect(void *socket, const char *endpoint);
void* zmq_ctx_new();
int zmq_ctx_destroy(void *context);
int zmq_ctx_get(void *context, int opt);
int zmq_ctx_set(void *context, int opt, int optval);
int zmq_proxy(void *frontend, void *backend, void *capture);
int zmq_proxy_steerable(void *frontend,
                        void *backend,
                        void *capture,
                        void *control);
int zmq_socket_monitor(void *socket, const char *addr, int events);

int zmq_curve_keypair (char *z85_public_key, char *z85_secret_key);
int zmq_curve_public (char *z85_public_key, char *z85_secret_key);
int zmq_has (const char *capability);

typedef struct { ...; } zmq_msg_t;
typedef ... zmq_free_fn;

int zmq_msg_init(zmq_msg_t *msg);
int zmq_msg_init_size(zmq_msg_t *msg, size_t size);
int zmq_msg_init_data(zmq_msg_t *msg,
                      void *data,
                      size_t size,
                      zmq_free_fn *ffn,
                      void *hint);

size_t zmq_msg_size(zmq_msg_t *msg);
void *zmq_msg_data(zmq_msg_t *msg);
int zmq_msg_close(zmq_msg_t *msg);

int zmq_msg_copy(zmq_msg_t *dst, zmq_msg_t *src);
int zmq_msg_send(zmq_msg_t *msg, void *socket, int flags);
int zmq_msg_recv(zmq_msg_t *msg, void *socket, int flags);

int zmq_getsockopt(void *socket,
                   int option_name,
                   void *option_value,
                   size_t *option_len);

int zmq_setsockopt(void *socket,
                   int option_name,
                   const void *option_value,
                   size_t option_len);

typedef int... ZMQ_FD_T;

typedef struct
{
    void *socket;
    ZMQ_FD_T fd;
    short events;
    short revents;
} zmq_pollitem_t;

int zmq_poll(zmq_pollitem_t *items, int nitems, long timeout);

// miscellany
void * memcpy(void *restrict s1, const void *restrict s2, size_t n);
void * malloc(size_t sz);
void free(void *p);
int get_ipc_path_max_len(void);

typedef struct _zhint {
    void *sock;
    void *mutex;
    size_t id;
} zhint;

typedef ... mutex_t;
mutex_t* mutex_allocate();

int zmq_wrap_msg_init_data(zmq_msg_t *msg,
                      void *data,
                      size_t size,
                      void *hint);
