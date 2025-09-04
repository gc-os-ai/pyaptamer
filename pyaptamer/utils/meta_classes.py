class NoNewPublicMethods(type):
    """A metaclass that prevents subclasses from:
    1. Adding new public methods.
    2. Overriding existing public methods.
    Use with `metaclass=NoNewPublicMethods`.
    """

    def __init__(cls, name, bases, namespace):
        # if the only base is object, allow free definition (root class)
        if all(base is object for base in bases):
            super().__init__(name, bases, namespace)
            return

        # collect all allowed public methods from bases
        allowed = {}
        for base in bases:
            for attr_name, attr_val in base.__dict__.items():
                if callable(attr_val) and not attr_name.startswith("_"):
                    allowed[attr_name] = attr_val

        # check the new class namespace
        for attr_name, attr_val in namespace.items():
            if callable(attr_val) and not attr_name.startswith("_"):
                if attr_name not in allowed:
                    raise TypeError(
                        f"Class {name} is not allowed to define new "
                        f"public method '{attr_name}'"
                    )
                elif attr_val is not allowed[attr_name]:
                    raise TypeError(
                        f"Class {name} is not allowed to override "
                        f"existing public method '{attr_name}'"
                    )

        super().__init__(name, bases, namespace)
