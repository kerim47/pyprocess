import textwrap

def warning(message):
    width = 60
    padding = 2
    box_width = width + 2 * padding
    
    print("!" * box_width)
    print("!" + " " * (box_width - 2) + "!")
    print("!" + " WARNING ".center(box_width - 2) + "!")
    print("!" + " " * (box_width - 2) + "!")
    
    wrapped_message = textwrap.wrap(message, width=width)
    for line in wrapped_message:
        print(f"!  {line.ljust(width)}!")
    
    print("!" + " " * (box_width - 2) + "!")
    print("!" * box_width)