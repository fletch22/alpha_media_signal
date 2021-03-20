import random
import string


def select_random_from_list(the_list: list, number_needed):
    if len(the_list) < number_needed:
        raise Exception("The number needed is greater than the size of the list")

    cloned = the_list.copy()

    items = []
    for i in range(number_needed):
        list_length = len(cloned)
        random_num = random.randrange(0, list_length)

        items.append(cloned.pop(random_num))

    return items


def get_random_string():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(10))