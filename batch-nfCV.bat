#!bin/bash
rm ./log/log-nfCV-512-10
echo '*********************  STARTS [CPU] PROCESSING  ********************************'>>./log/log-nfCV-512-10
python3 BPARmainCV.py --num-processes 2  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 2 RUNNING 1st REPEAT'>>./log/log-nfCV-512-10
python3 BPARmainCV.py --num-processes 2  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 2 RUNNING 2nd REPEAT'>>./log/log-nfCV-512-10
python3 BPARmainCV.py --num-processes 2  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 2 RUNNING 3rd REPEAT'>>./log/log-nfCV-512-10
python3 BPARmainCV.py --num-processes 4  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 4 RUNNING 1st REPEAT'>>./log/log-nfCV-512-10
python3 BPARmainCV.py --num-processes 4  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 4 RUNNING 2nd REPEAT'>>./log/log-nfCV-512-10
python3 BPARmainCV.py --num-processes 4  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 4 RUNNING 3rd REPEAT'>>./log/log-nfCV-512-10
python3 BPARmainCV.py --num-processes 8  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 8 RUNNING 1st REPEAT'>>./log/log-nfCV-512-10
python3 BPARmainCV.py --num-processes 8  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 8 RUNNING 2nd REPEAT'>>./log/log-nfCV-512-10
python3 BPARmainCV.py --num-processes 8  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 8 RUNNING 3rd REPEAT'>>./log/logCPU-1-8-512-10
echo '*********************  STARTS [GPU] PROCESSING  ********************************'>>./log/log-nfCV-512-10
python3 GPUmainLeNet.py --cuda --num-processes 1  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 1 RUNNING 1st REPEAT'>>./log/log-nfCV-512-10
python3 GPUmainLeNet.py --cuda --num-processes 1  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 1 RUNNING 2nd REPEAT'>>./log/log-nfCV-512-10
python3 GPUmainLeNet.py --cuda --num-processes 1  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 1 RUNNING 3rd REPEAT'>>./log/log-nfCV-512-10
python3 GPUmainLeNet.py --cuda --num-processes 2  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 2 RUNNING 1st REPEAT'>>./log/log-nfCV-512-10
python3 GPUmainLeNet.py --cuda --num-processes 2  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 2 RUNNING 2nd REPEAT'>>./log/log-nfCV-512-10
python3 GPUmainLeNet.py --cuda --num-processes 2  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 2 RUNNING 3rd REPEAT'>>./log/log-nfCV-512-10
python3 GPUmainLeNet.py --cuda --num-processes 4  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 4 RUNNING 1st REPEAT'>>./log/log-nfCV-512-10
python3 GPUmainLeNet.py --cuda --num-processes 4  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 4 RUNNING 2nd REPEAT'>>./log/log-nfCV-512-10
python3 GPUmainLeNet.py --cuda --num-processes 4  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 4 RUNNING 3rd REPEAT'>>./log/log-nfCV-512-10
python3 GPUmainLeNet.py --cuda --num-processes 8  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 8 RUNNING 1st REPEAT'>>./log/log-nfCV-512-10
python3 GPUmainLeNet.py --cuda --num-processes 8  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 8 RUNNING 2nd REPEAT'>>./log/log-nfCV-512-10
python3 GPUmainLeNet.py --cuda --num-processes 8  --batch-size 512 --epochs 10 >>./log/log-nfCV-512-10
echo '!@@END 8 RUNNING 3rd REPEAT'>>./log/log-nfCV-512-10
echo '**********END OF PROCESS********'
