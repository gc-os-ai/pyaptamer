class NoNewPublicMethods(type):
    """A metaclass that prevents the addition of new public methods in subclasses.
    Use using `metaclass=NoNewPublicMethods` in your class definition."""

    def __init__(cls, name, bases, namespace):
        # Collect all public methods already defined in base classes
        allowed = set()
        for base in bases:
            for attr_name, attr_val in base.__dict__.items():
                if callable(attr_val) and not attr_name.startswith("_"):
                    allowed.add(attr_name)

        # Check new class namespace for new public methods
        for attr_name, attr_val in namespace.items():
            if callable(attr_val) and not attr_name.startswith("_"):
                if attr_name not in allowed:
                    raise TypeError(
                        "Class {name} is not allowed to define new"
                        f"public method '{attr_name}'"
                    )

        super().__init__(name, bases, namespace)
