import pytest


def rotate_90(matrix):
    return [list(r) for r in zip(*matrix)]


class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        snake = []

        while True:
            if len(matrix) > 0:
                # Top row
                snake += matrix[0]

                # remove top row.
                matrix = matrix[1:]

                if len(matrix) == 0:
                    break

                # get last elements of remain rows
                # remove last elements of remaining rows
                new_matrix = []
                for row in matrix:
                    if len(row) > 0:
                        snake.append(row[-1])
                        row = row[:-1]
                        row = row[::-1]
                        new_matrix.append(row)
                matrix = new_matrix

                # reverse the array and continue
                matrix = matrix[::-1]
            else:
                break

        return snake


@pytest.mark.parametrize(
    "matrix,expected", [
        [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]],
        [[[1]], [1]],
        [[[2,3]], [2, 3]],
        [[[1,2,3],[4,5,6],[7,8,9]], [1, 2, 3, 6, 9, 8, 7, 4, 5]],
        [[[3],[2]], [3, 2]],
        [[[7],[9],[6]], [7,9,6]]
    ]
)
def test_spiral(matrix, expected):
    # Arrange
    s = Solution()

    result = s.spiralOrder(matrix)

    print(result)

    assert(result == expected)