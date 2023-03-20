from typing import Generic, TypeVar, Tuple

Inputs = TypeVar("")


Inputs = TypeVar('Inputs')
Output = TypeVar('Output')

class MyCallable(Generic[Inputs, Outputs]):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args: Inputs) -> Output:
        return self.func(*args)

def my_function(arg1: int, arg2: str) -> float:
    return float(arg1 + len(arg2))

# my_callable: MyCallable[Tuple[int, str], float] = MyCallable(my_function)
# result = my_callable(42, 'hello')


from typing import TypeVar, Tuple

Inputs = TypeVar('Inputs', bound=Tuple)

def my_function(inputs: Inputs) -> None:
    print(inputs)

# my_function((1, 'hello'))  # OK
# my_function([1, 'hello'])  # Error: List is not a subtype of Tuple



from typing import TypeVar, Union

MyType = TypeVar('MyType', int, str)

def my_function(arg: MyType) -> None:
    print(arg)

# my_function(42)  # OK
# my_function('hello')  # OK
# my_function(3.14)  # Error: float is not in the set of accepted types





from typing import NewType

class Space:
    Point = NewType('Point', tuple)
    Manifold = NewType('Manifold', object)

def my_function(point: Space.Point) -> None:
    print(point)

# my_point = (1, 2)
# my_function(my_point)