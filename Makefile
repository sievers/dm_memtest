CC=gcc
INCLUDES=-I.



CFLAGS=-std=c99 -g -march=native -fopenmp -O3 ${INCLUDES}
CLIBS=-lgomp


%.o:%.c
	$(CC) $(CFLAGS) -c $< -o $@

CPROGS = test_dram_speed simple_dedisperse_block_threaded simple_dedisperse_block_threaded_memtest

all: ${CPROGS}

test_dram_speed: test_dram_speed.o nt_memcpy.o
	$(CC) $(CFLAGS) test_dram_speed.o nt_memcpy.o -o $@  $(CLIBS) 


simple_dedisperse_block_threaded: simple_dedisperse_block_threaded.o
	$(CC) $(CFLAGS) simple_dedisperse_block_threaded.o -o $@  $(CLIBS) 

simple_dedisperse_block_threaded_memtest: simple_dedisperse_block_threaded_memtest.o nt_memcpy.o 
	$(CC) $(CFLAGS) simple_dedisperse_block_threaded_memtest.o nt_memcpy.o -o $@  $(CLIBS)

clean:
	rm -f *.o
	rm -f $(CPROGS)
