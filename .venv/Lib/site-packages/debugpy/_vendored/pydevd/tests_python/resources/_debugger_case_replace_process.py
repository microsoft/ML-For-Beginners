import sys

if __name__ == '__main__':
    if 'in-sub' not in sys.argv:
        import os
        # These functions all execute a new program, replacing the current process; they do not return.
        # os.execl(path, arg0, arg1, ...)
        # os.execle(path, arg0, arg1, ..., env)
        # os.execlp(file, arg0, arg1, ...)
        # os.execlpe(file, arg0, arg1, ..., env)¶
        # os.execv(path, args)
        # os.execve(path, args, env)
        # os.execvp(file, args)
        # os.execvpe(file, args, env)
        os.execvp(sys.executable, [sys.executable, __file__, 'in-sub'])
    else:
        print('In sub')
        print('TEST SUCEEDED!')
