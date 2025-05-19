import sys
import argparse
import random
import csv
from faker import Faker

def generate_integers(count, min_val, max_val, output_file):
    numbers = [str(random.randint(min_val, max_val)) for _ in range(count)]
    with open(output_file, "w") as f:
        f.write(",".join(numbers))

def generate_words(count, output_file):
    fake = Faker()
    words = [fake.word() for _ in range(count)]
    with open(output_file, "w") as f:
        f.write(" ".join(words))

def generate_matrix(width, height, output_file):
    matrix = [[random.randint(1, 100) for _ in range(width)] for _ in range(height)]
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(matrix)

def main():
    parser = argparse.ArgumentParser(description="Data generator script")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Integer data
    int_parser = subparsers.add_parser("int", help="Generate integer data")
    int_parser.add_argument("--count", type=int, required=True, help="Number of integers")
    int_parser.add_argument("--min", type=int, required=True, help="Minimum integer value")
    int_parser.add_argument("--max", type=int, required=True, help="Maximum integer value")
    int_parser.add_argument("--output", type=str, default="random_numbers.txt", help="Output file name")

    # Word data
    word_parser = subparsers.add_parser("words", help="Generate fake words")
    word_parser.add_argument("--count", type=int, required=True, help="Number of words")
    word_parser.add_argument("--output", type=str, default="random_words.txt", help="Output file name")

    # Matrix data
    matrix_parser = subparsers.add_parser("matrix", help="Generate a random integer matrix")
    matrix_parser.add_argument("--width", type=int, required=True, help="Matrix width")
    matrix_parser.add_argument("--height", type=int, required=True, help="Matrix height")
    matrix_parser.add_argument("--output", type=str, default="matrix.csv", help="Output CSV file name")

    args = parser.parse_args()

    if args.mode == "int":
        generate_integers(args.count, args.min, args.max, args.output)
    elif args.mode == "words":
        generate_words(args.count, args.output)
    elif args.mode == "matrix":
        generate_matrix(args.width, args.height, args.output)

if __name__ == "__main__":
    # # print usage 
    # print("Usage:")
    # print("  python script.py int --count N --min MIN --max MAX [--output FILE]")
    # print("  python script.py words --count N [--output FILE]")
    # print("  python script.py matrix --width W --height H [--output FILE]")
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python3 script.py int --count N --min MIN --max MAX [--output FILE]")
        print("  python3 script.py words --count N [--output FILE]")
        print("  python3 script.py matrix --width W --height H [--output FILE]")
        sys.exit(1)

    main()