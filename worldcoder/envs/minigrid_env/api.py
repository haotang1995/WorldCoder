class Entity:
    def __init__(self, x, y, **kwargs):
        self.name = self.__class__.__name__
        self.x = x
        self.y = y
        for key, value in kwargs.items():
            setattr(self, key, value)
    def __repr__(self):
        attr = ', '.join(f'{key}={value}' for key, value in self.__dict__.items() if key not in ('name', 'x', 'y'))
        if attr: return f"{self.name}({self.x}, {self.y}, {attr})"
        else: return f"{self.name}({self.x}, {self.y})"
    def __eq__(self, other):
        return all(getattr(self, key) == getattr(other, key, None) for key in self.__dict__.keys())
    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))
class Agent(Entity): pass
class Key(Entity): pass
class Door(Entity): pass
class Goal(Entity): pass
class Wall(Entity): pass
class Box(Entity): pass
class Ball(Entity): pass
class Lava(Entity): pass
def get_entities_by_name(entities, name):
    return [ entity for entity in entities if entity.name == name ]
def get_entities_by_position(entities, x, y):
    return [ entity for entity in entities if entity.x == x and entity.y == y ]


