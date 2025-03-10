import random

import random

def get_test_values(filename, value, lower_bound, upper_bound):
    z_values = []
    frequency_values = []
    with open(filename, "r") as file:
        line_count = 0
        for line in file:
            parts = line.strip().split("\t")

            frequency_values.append(float(parts[0]))
            z_values.append(float(parts[1]))
            line_count += 1

            if random.random() < (line_count / value):
                skip_count = random.randint(lower_bound, upper_bound)
                for _ in range(skip_count):
                    if not next(file, None):
                        break

    return z_values, frequency_values

def create_test(filename, value, lower_bound=50, upper_bound=1000):
    z, f = get_test_values(filename, value, lower_bound, upper_bound)
    file_name = filename.split(".")[0]
    with open(f"{file_name}_test", "w") as file:
        for z, f in zip(z, f):
            file.write(f"{f}\t{z}\n")

if __name__ == "__main__":
    create_test("no_shoes_1.txt", 1000, 50, 500)
    create_test("no_shoes_2.txt", 1000, 50, 500)
    create_test("no_shoes_3.txt", 1000, 50, 500)
    create_test("no_shoes_6.txt", 1000, 50, 500)
    create_test("no_shoes_9.txt", 1000, 50, 500)
    create_test("no_shoes_12.txt", 1000, 50, 500)
