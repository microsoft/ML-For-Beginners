import subprocess
import sys
print(sys.executable)

if __name__ == '__main__':
    p = subprocess.Popen([sys.executable, '-u', '_always_live_program.py'])
    import attach_pydevd
    attach_pydevd.main(attach_pydevd.process_command_line(['--pid', str(p.pid), '--protocol', 'http']))
    p.wait()
