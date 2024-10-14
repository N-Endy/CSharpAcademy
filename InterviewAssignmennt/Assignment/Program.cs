// TASK 1

string name = "John Doe";
int age = 25;
bool isAdmin = true;

string message = $"Name: {name}\nAge: {age}\nisAdmin: {isAdmin}";

Console.WriteLine(message);

// TASK 2

Console.Write("Enter any number greater than zero: ");
int userInput;
while (!int.TryParse(Console.ReadLine(), out userInput) || userInput <= 0)
{
    Console.WriteLine("The number you entered is less than or equal to zero");
    Console.Write("Please enter a number greater than zero: ");
}

if (userInput % 2 == 0)
    Console.WriteLine("Even");
else
    Console.WriteLine("Odd");


// TASK 3

for (var i = 1; i <= 10; i++)
{
    Console.Write($"{i} ");
}
Console.WriteLine("");


// TASK 4
int[] numberArray = [2, 4, 6, 8];
var sum = numberArray.Sum();
Console.WriteLine($"Sum of numbers in numberArray is {sum}");

foreach (var number in numberArray)
{
    Console.WriteLine(number);
}


// TASK 5
static void Greet(string name)
{
    Console.WriteLine($"Hello, {name}!");
}

Greet("Alice");