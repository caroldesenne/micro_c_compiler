all: semantic gencode

uc:
	python3 uc.py tests/t1.uc
	python3 uc.py tests/t2.uc
	python3 uc.py tests/t3.uc
	python3 uc.py tests/t4.uc
	python3 uc.py tests/t5.uc
	python3 uc.py tests/armstrong.uc
	python3 uc.py tests/bubble.uc
	python3 uc.py tests/factorial.uc
	python3 uc.py tests/gdc.uc
	python3 uc.py tests/primes.uc
	#python3 uc.py tests/ptrFuncion.uc
	python3 uc.py tests/ir1.uc
	python3 uc.py tests/ir2.uc
	python3 uc.py tests/ir3.uc
	python3 uc.py tests/ir4.uc
	python3 uc.py tests/ir5.uc
	python3 uc.py tests/ir6.uc
	python3 uc.py tests/ir7.uc
	python3 uc.py tests/ir8.uc
	python3 uc.py tests/ir9.uc

clean:
	rm parser.out parsetab.py

semantic:
	python3 check.py tests/t1.uc
	python3 check.py tests/t2.uc
	python3 check.py tests/t3.uc
	python3 check.py tests/t4.uc
	python3 check.py tests/t5.uc
	python3 check.py tests/test1.uc
	python3 check.py tests/test2.uc
	#python3 check.py tests/test3.uc
	python3 check.py tests/test4.uc
	#python3 check.py tests/test5.uc
	#python3 check.py tests/test6.uc
	python3 check.py tests/test7.uc
	#python3 check.py tests/test8.uc
	python3 check.py tests/armstrong.uc
	python3 check.py tests/arrayRef.uc
	python3 check.py tests/bubble.uc
	python3 check.py tests/charArray.uc
	python3 check.py tests/factorial.uc
	python3 check.py tests/gdc.uc
	python3 check.py tests/primes.uc
	#python3 check.py tests/ptrFunction.uc

gencode:
	python3 generate.py tests/ir1.uc
	python3 generate.py tests/ir2.uc
	python3 generate.py tests/ir3.uc
	python3 generate.py tests/ir4.uc
	python3 generate.py tests/ir5.uc
	python3 generate.py tests/ir6.uc
	python3 generate.py tests/ir7.uc
	python3 generate.py tests/ir8.uc
	python3 generate.py tests/ir9.uc
	-diff tests/ir1.out tests/ir1.code
	-diff tests/ir2.out tests/ir2.code
	-diff tests/ir3.out tests/ir3.code
	-diff tests/ir4.out tests/ir4.code
	-diff tests/ir5.out tests/ir5.code
	-diff tests/ir6.out tests/ir6.code
	-diff tests/ir7.out tests/ir7.code
	-diff tests/ir8.out tests/ir8.code
	-diff tests/ir9.out tests/ir9.code
