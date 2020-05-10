all: semantic gencode

uc: uc_commands

uc_commands:
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
	python3 uc.py tests/ir1.uc
	python3 uc.py tests/ir2.uc
	python3 uc.py tests/ir3.uc
	python3 uc.py tests/ir4.uc
	python3 uc.py tests/ir5.uc
	python3 uc.py tests/ir6.uc
	python3 uc.py tests/ir7.uc
	python3 uc.py tests/ir8.uc
	python3 uc.py tests/ir9.uc
	python3 uc.py tests/susy11.uc
	python3 uc.py tests/susy12.uc
	python3 uc.py tests/susy19.uc
	python3 uc.py tests/susy19_.uc

clean:
	-rm parser.out parsetab.py
	-rm tests/*.ast
	-rm tests/*.code
	-rm tests/*.ir

semantic:
	python3 checker.py tests/t1.uc
	python3 checker.py tests/t2.uc
	python3 checker.py tests/t3.uc
	python3 checker.py tests/t4.uc
	python3 checker.py tests/t5.uc
	python3 checker.py tests/test1.uc
	python3 checker.py tests/test2.uc
	#python3 checker.py tests/test3.uc
	python3 checker.py tests/test4.uc
	#python3 checker.py tests/test5.uc
	#python3 checker.py tests/test6.uc
	python3 checker.py tests/test7.uc
	#python3 checker.py tests/test8.uc
	python3 checker.py tests/armstrong.uc
	python3 checker.py tests/arrayRef.uc
	python3 checker.py tests/bubble.uc
	python3 checker.py tests/charArray.uc
	python3 checker.py tests/factorial.uc
	python3 checker.py tests/gdc.uc
	python3 checker.py tests/primes.uc
	#python3 checker.py tests/ptrFunction.uc

gencode:
	python3 generator.py tests/ir1.uc
	python3 generator.py tests/ir2.uc
	python3 generator.py tests/ir3.uc
	python3 generator.py tests/ir4.uc
	python3 generator.py tests/ir5.uc
	python3 generator.py tests/ir6.uc
	python3 generator.py tests/ir7.uc
	python3 generator.py tests/ir8.uc
	python3 generator.py tests/ir9.uc
	#-diff tests/ir1.out tests/ir1.code
	#-diff tests/ir2.out tests/ir2.code
	#-diff tests/ir3.out tests/ir3.code
	#-diff tests/ir4.out tests/ir4.code
	#-diff tests/ir5.out tests/ir5.code
	#-diff tests/ir6.out tests/ir6.code
	#-diff tests/ir7.out tests/ir7.code
	#-diff tests/ir8.out tests/ir8.code
	#-diff tests/ir9.out tests/ir9.code
