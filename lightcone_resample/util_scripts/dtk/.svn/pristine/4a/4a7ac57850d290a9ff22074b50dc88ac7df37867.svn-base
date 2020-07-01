CC = mpicxx
CFLAGS  = -I../dtk/. -I../genericio -g
LDFLAGS = -L../lib/. -L../genericio -ldtk -lfftw3f -lfftw3_threads -lfftw3 -lm -lpthread

.PHONY: test
test: main.cpp 
	make -C ../.
	${CC} ${CFLAGS} -o $@ $^ ${LDFLAGS}
.PHONY:
clean:
	rm test
