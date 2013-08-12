all: libeigenmat.so

libeigenmat.so: eigenmat.cc eigenmat.h ziggurat.cpp ziggurat.hpp
	g++ -O3 -fPIC -I/u/nitish/eigen -o libeigenmat.so --shared -fopenmp eigenmat.cc ziggurat.cpp

clean:
	rm -rf *.linkinfo *.pyc *.so
