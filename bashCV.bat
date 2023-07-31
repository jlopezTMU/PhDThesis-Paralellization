#!bin/bash
rm ./log/logCV-512-10
echo '*********************  STARTS C P U PROCESSING  ********************************'>>./log/logCV-512-10
python3 fusedCV2.py --num-processes 1  --batch-size 64 --epochs 5 >./log/logCV-512-10
echo '!@@END 1 RUNNING CPU 1st REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --num-processes 1  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 1 RUNNING CPU 2nd REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --num-processes 1  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 1 RUNNING CPU 3rd REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --num-processes 2  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 2 RUNNING CPU 1st REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --num-processes 2  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 2 RUNNING CPU 2nd REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --num-processes 2  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 2 RUNNING CPU 3rd REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --num-processes 4  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 4 RUNNING CPU 1st REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --num-processes 4  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 4 RUNNING CPU 2nd REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --num-processes 4  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 4 RUNNING CPU 3rd REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --num-processes 8  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 8 RUNNING CPU 1st REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --num-processes 8  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 8 RUNNING CPU 2nd REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --num-processes 8  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 8 RUNNING CPU 3rd REPEAT'>>./log/logCPU-1-8-64-5
echo '*********************  STARTS G P U PROCESSING  ********************************'>>./log/logCV-512-10
python3 fusedCV2.py --cuda --num-processes 1  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 1 RUNNING GPU 1st REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --cuda --num-processes 1  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 1 RUNNING GPU 2nd REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --cuda --num-processes 1  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 1 RUNNING GPU 3rd REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --cuda --num-processes 2  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 2 RUNNING GPU 1st REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --cuda --num-processes 2  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 2 RUNNING GPU 2nd REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --cuda --num-processes 2  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 2 RUNNING GPU 3rd REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --cuda --num-processes 4  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 4 RUNNING GPU 1st REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --cuda --num-processes 4  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 4 RUNNING GPU 2nd REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --cuda --num-processes 4  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 4 RUNNING GPU 3rd REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --cuda --num-processes 8  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 8 RUNNING GPU 1st REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --cuda --num-processes 8  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 8 RUNNING GPU 2nd REPEAT'>>./log/logCV-512-10
python3 fusedCV2.py --cuda --num-processes 8  --batch-size 64 --epochs 5 >>./log/logCV-512-10
echo '!@@END 8 RUNNING GPU 3rd REPEAT'>>./log/logCV-512-10
echo 'END OF PROCESS'
