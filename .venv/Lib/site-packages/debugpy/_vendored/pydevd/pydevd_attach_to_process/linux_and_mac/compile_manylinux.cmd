:: WARNING: manylinux1 images are based on CentOS 5, which requires vsyscall to be available on
:: the host. For any recent version of Linux, this requires passing vsyscall=emulate during boot.
:: For WSL, add the following to your .wslconfig:
::
::   [wsl2]
::   kernelCommandLine = vsyscall=emulate

docker run --rm -v %~dp0/..:/src quay.io/pypa/manylinux1_x86_64 g++ -std=c++11 -shared -o /src/attach_linux_amd64.so -fPIC -nostartfiles /src/linux_and_mac/attach.cpp

docker run --rm -v %~dp0/..:/src quay.io/pypa/manylinux1_i686 g++ -std=c++11 -shared -o /src/attach_linux_x86.so -fPIC -nostartfiles /src/linux_and_mac/attach.cpp
