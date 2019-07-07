import sys
if __name__ == "__main__":
    path = sys.argv[1]
    m = int(sys.argv[2])
    with open(path, 'r') as the_file:
        lines = the_file.readlines()
    n = len(lines)
    assert m <= n
    with open(path + "_TRUNC_" + str(m), 'a') as new_file:
        for line in lines[:m]:
            new_file.write(line + '\n')
