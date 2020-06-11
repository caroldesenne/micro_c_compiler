all: semantic gencode

uc: uc_commands

uc_commands:
	python3 uc.py ../tests/t1.uc
	python3 uc.py ../tests/t2.uc
	python3 uc.py ../tests/t3.uc
	python3 uc.py ../tests/t4.uc
	python3 uc.py ../tests/t5.uc
	python3 uc.py ../tests/armstrong.uc
	python3 uc.py ../tests/bubble.uc
	python3 uc.py ../tests/factorial.uc
	python3 uc.py ../tests/gdc.uc
	python3 uc.py ../tests/primes.uc
	python3 uc.py ../tests/ir1.uc
	python3 uc.py ../tests/ir2.uc
	python3 uc.py ../tests/ir3.uc
	python3 uc.py ../tests/ir4.uc
	python3 uc.py ../tests/ir5.uc
	python3 uc.py ../tests/ir6.uc
	python3 uc.py ../tests/ir7.uc
	python3 uc.py ../tests/ir8.uc
	python3 uc.py ../tests/ir9.uc
	python3 uc.py ../tests/susy11.uc
	python3 uc.py ../tests/susy12.uc
	python3 uc.py ../tests/susy19.uc
	python3 uc.py ../tests/susy19_.uc

clean:
	-rm parser.out parsetab.py
	-rm ../tests/*.ast
	-rm ../tests/*.code
	-rm ../tests/*.ir

optimization:
	python3 optimizer.py ../tests/armstrong.uc
	python3 optimizer.py ../tests/arrayRef.uc
	python3 optimizer.py ../tests/bubble.uc
	python3 optimizer.py ../tests/charArray.uc
	python3 optimizer.py ../tests/factorial.uc
	python3 optimizer.py ../tests/foo.uc
	python3 optimizer.py ../tests/gdc.uc
	python3 optimizer.py ../tests/imseven.uc
	python3 optimizer.py ../tests/initializeArray.uc
	python3 optimizer.py ../tests/ir1.uc
	python3 optimizer.py ../tests/ir2.uc
	python3 optimizer.py ../tests/ir3.uc
	python3 optimizer.py ../tests/ir4.uc
	python3 optimizer.py ../tests/ir5.uc
	python3 optimizer.py ../tests/ir6.uc
	python3 optimizer.py ../tests/ir7.uc
	python3 optimizer.py ../tests/ir8.uc
	python3 optimizer.py ../tests/ir9.uc
	python3 optimizer.py ../tests/la1.uc
	python3 optimizer.py ../tests/la2.uc
	python3 optimizer.py ../tests/la3.uc
	python3 optimizer.py ../tests/la4.uc
	python3 optimizer.py ../tests/la5.uc
	python3 optimizer.py ../tests/primes.uc
	python3 optimizer.py ../tests/rd1.uc
	python3 optimizer.py ../tests/susy11.uc
	python3 optimizer.py ../tests/susy12.uc
	python3 optimizer.py ../tests/susy19.uc
	python3 optimizer.py ../tests/susy19_.uc
	python3 optimizer.py ../tests/t1.uc
	python3 optimizer.py ../tests/t10.uc
	python3 optimizer.py ../tests/t11.uc
	python3 optimizer.py ../tests/t12.uc
	python3 optimizer.py ../tests/t13.uc
	python3 optimizer.py ../tests/t14.uc
	python3 optimizer.py ../tests/t15.uc
	python3 optimizer.py ../tests/t16.uc
	python3 optimizer.py ../tests/t17.uc
	python3 optimizer.py ../tests/t18.uc
	python3 optimizer.py ../tests/t19.uc
	python3 optimizer.py ../tests/t2.uc
	python3 optimizer.py ../tests/t20.uc
	python3 optimizer.py ../tests/t21.uc
	python3 optimizer.py ../tests/t3.uc
	python3 optimizer.py ../tests/t4.uc
	python3 optimizer.py ../tests/t5.uc
	python3 optimizer.py ../tests/t6.uc
	python3 optimizer.py ../tests/t7.uc
	python3 optimizer.py ../tests/t8.uc
	python3 optimizer.py ../tests/t9.uc

semantic:
	python3 checker.py ../tests/t1.uc
	python3 checker.py ../tests/t2.uc
	python3 checker.py ../tests/t3.uc
	python3 checker.py ../tests/t4.uc
	python3 checker.py ../tests/t5.uc
	python3 checker.py ../tests/test1.uc
	python3 checker.py ../tests/test2.uc
	#python3 checker.py ../tests/test3.uc
	python3 checker.py ../tests/test4.uc
	#python3 checker.py ../tests/test5.uc
	#python3 checker.py ../tests/test6.uc
	python3 checker.py ../tests/test7.uc
	#python3 checker.py ../tests/test8.uc
	python3 checker.py ../tests/armstrong.uc
	python3 checker.py ../tests/arrayRef.uc
	python3 checker.py ../tests/bubble.uc
	python3 checker.py ../tests/charArray.uc
	python3 checker.py ../tests/factorial.uc
	python3 checker.py ../tests/gdc.uc
	python3 checker.py ../tests/primes.uc
	#python3 checker.py ../tests/ptrFunction.uc

gencode:
	python3 generator.py ../tests/ir1.uc
	python3 generator.py ../tests/ir2.uc
	python3 generator.py ../tests/ir3.uc
	python3 generator.py ../tests/ir4.uc
	python3 generator.py ../tests/ir5.uc
	python3 generator.py ../tests/ir6.uc
	python3 generator.py ../tests/ir7.uc
	python3 generator.py ../tests/ir8.uc
	python3 generator.py ../tests/ir9.uc
	#-diff ../tests/ir1.out ../tests/ir1.code
	#-diff ../tests/ir2.out ../tests/ir2.code
	#-diff ../tests/ir3.out ../tests/ir3.code
	#-diff ../tests/ir4.out ../tests/ir4.code
	#-diff ../tests/ir5.out ../tests/ir5.code
	#-diff ../tests/ir6.out ../tests/ir6.code
	#-diff ../tests/ir7.out ../tests/ir7.code
	#-diff ../tests/ir8.out ../tests/ir8.code
	#-diff ../tests/ir9.out ../tests/ir9.code
