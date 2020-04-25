all:
	python3 uc.py tests/t1.uc
	python3 uc.py tests/t2.uc
	python3 uc.py tests/t3.uc
	python3 uc.py tests/t4.uc
	python3 uc.py tests/t5.uc
	-diff tests/t1.ast tests/out1.ast
	-diff tests/t2.ast tests/out2.ast
	-diff tests/t3.ast tests/out3.ast
	-diff tests/t4.ast tests/out4.ast
	-diff tests/t5.ast tests/out5.ast
	python3 uc.py tests/armstrong.uc
	python3 uc.py tests/bubble.uc
	python3 uc.py tests/factorial.uc
	python3 uc.py tests/gdc.uc
	python3 uc.py tests/primes.uc
	python3 uc.py tests/ptrFuncion.uc

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
