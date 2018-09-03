import requests
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4)
def classify(frame, frame_no):

    # Send to API
    r = requests.post('http://localhost:8881/classify',
        json={'image': frame, 'frame_no': frame_no})

    # Print result from API
    pp.pprint(r.json())
    return r.json()

def main():
    retval = classify('Example frame', 10)

if __name__ == "__main__":
    main()