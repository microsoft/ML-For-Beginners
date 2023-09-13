set -e
SRC="$(dirname "$0")/.."
g++ -fPIC -D_REENTRANT -std=c++11 -arch x86_64 -c $SRC/linux_and_mac/attach.cpp -o $SRC/attach_x86_64.o
g++ -dynamiclib -nostartfiles -arch x86_64 -lc $SRC/attach_x86_64.o -o $SRC/attach_x86_64.dylib
