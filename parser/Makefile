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
