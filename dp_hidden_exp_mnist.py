import subprocess

if __name__ == '__main__':
    epoch = 15
    batch_size = 512
    for seed in [22, 99, 1]:
        for sparsity in [0.1, 0.3, 0.5, 0.7]:
            for dp in [True, False]:
                print(f"{'DP ' if dp else ''}sparsity={sparsity}, seed={seed}")
                # subprocess.run(f"python simple_cifar_example_dp.py --epochs {epoch} --log-interval -1 --batch-size {batch_size} --sparsity {sparsity} --seed {seed}{' --no-dp' if not dp else ''}")
                subprocess.run(f"python simple_mnist_example_dp.py --epochs {epoch} --log-interval -1 --batch-size {batch_size} --sparsity {sparsity} --seed {seed}{' --no-dp' if not dp else ''}")