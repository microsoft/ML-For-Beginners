
class Car(object):
    """A car class"""
    def __init__(self, model, make, color):
        self.model = model
        self.make = make
        self.color = color
        self.price = None

    def get_price(self):
        return self.price

    def set_price(self, value):
        self.price = value

availableCars = []
def main():
    global availableCars

    #Create a new car obj
    carObj = Car("Maruti SX4", "2011", "Black")
    carObj.set_price(950000)  # Set price
    # Add this to available cars
    availableCars.append(carObj)

    print('TEST SUCEEDED')

if __name__ == '__main__':
    main()
