(python-language-reference)=

# Python Language Reference Coverage

This is a 1:1 mapping of the features listed in https://docs.python.org/3/reference/ and their
support in TorchScript. The categorizations are as follows:

```{list-table}
:widths: 40 40 20
:header-rows: 1

* - Section
  - Status
  - Note
* - [1. Introduction](https://docs.python.org/3/reference/introduction.html)
  - Not Relevant
  -
* - [1.1. Alternate Implementations](https://docs.python.org/3/reference/introduction.html#alternate-implementations)
  - Not Relevant
  -
* - [1.2. Notation](https://docs.python.org/3/reference/introduction.html#notation)
  - Not Relevant
  -
* - [2. Lexical analysis](https://docs.python.org/3/reference/lexical_analysis.html#)
  - Not Relevant
  -
* - [2.1. Line structure](https://docs.python.org/3/reference/lexical_analysis.html#line-structure)
  - Not Relevant
  -
* - [2.1.1. Logical lines](https://docs.python.org/3/reference/lexical_analysis.html#logical-lines)
  - Not Relevant
  -
* - [2.1.2. Physical lines](https://docs.python.org/3/reference/lexical_analysis.html#physical-lines)
  - Supported
  -
* - [2.1.3. Comments](https://docs.python.org/3/reference/lexical_analysis.html#comments)
  - Supported
  -
* - [2.1.4. Encoding declarations](https://docs.python.org/3/reference/lexical_analysis.html#encoding-declarations)
  - Not Supported
  - TorchScript explicitly don't support unicode
* - [2.1.5. Explicit line joining](https://docs.python.org/3/reference/lexical_analysis.html#explicit-line-joining)
  - Supported
  -
* - [2.1.6. Implicit line joining](https://docs.python.org/3/reference/lexical_analysis.html#implicit-line-joining)
  - Supported
  -
* - [2.1.7. Blank lines](https://docs.python.org/3/reference/lexical_analysis.html#blank-lines)
  - Supported
  -
* - [2.1.8. Indentation](https://docs.python.org/3/reference/lexical_analysis.html#indentation)
  - Supported
  -
* - [2.1.9. Whitespace between tokens](https://docs.python.org/3/reference/lexical_analysis.html#whitespace-between-tokens)
  - Not Relevant
  -
* - [2.2. Other tokens](https://docs.python.org/3/reference/lexical_analysis.html#other-tokens)
  - Not Relevant
  -
* - [2.3. Identifiers and keywords](https://docs.python.org/3/reference/lexical_analysis.html#identifiers)
  - Supported
  -
* - [2.3.1. Keywords](https://docs.python.org/3/reference/lexical_analysis.html#keywords)
  - Supported
  -
* - [2.3.2. Reserved classes of identifiers](https://docs.python.org/3/reference/lexical_analysis.html#reserved-classes-of-identifiers)
  - Supported
  -
* - [2.4. Literals](https://docs.python.org/3/reference/lexical_analysis.html#literals)
  - Not Relevant
  -
* - [2.4.1. String and Bytes literals](https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals)
  - Supported
  -
* - [2.4.2. String literal concatenation](https://docs.python.org/3/reference/lexical_analysis.html#string-literal-concatenation)
  - Supported
  -
* - [2.4.3. Formatted string literals](https://docs.python.org/3/reference/lexical_analysis.html#formatted-string-literals)
  - Partially Supported
  -
* - [2.4.4. Numeric literals](https://docs.python.org/3/reference/lexical_analysis.html#numeric-literals)
  - Supported
  -
* - [2.4.5. Integer literals](https://docs.python.org/3/reference/lexical_analysis.html#integer-literals)
  - Supported
  -
* - [2.4.6. Floating point literals](https://docs.python.org/3/reference/lexical_analysis.html#floating-point-literals)
  - Supported
  -
* - [2.4.7. Imaginary literals](https://docs.python.org/3/reference/lexical_analysis.html#imaginary-literals)
  - Not Supported
  -
* - [2.5. Operators](https://docs.python.org/3/reference/lexical_analysis.html#operators)
  - Partially Supported
  - Not supported: ``<<``, ``>>``, ``:=``
* - [2.6. Delimiters](https://docs.python.org/3/reference/lexical_analysis.html#delimiters)
  - Partially Supported
  - Not supported: ``**=``, ``<<=``, ``>>=``, ``%=``, ``^=``, ``@=``, ``&=``, ``//=``, ``%`` operator for some types (e.g. ``str``\ )
* - [3. Data model](https://docs.python.org/3/reference/datamodel.html#)
  - Not Relevant
  -
* - [3.1. Objects, values and types](https://docs.python.org/3/reference/datamodel.html#objects-values-and-types)
  - Not Relevant
  -
* - [3.2. The standard type hierarchy](https://docs.python.org/3/reference/datamodel.html#the-standard-type-hierarchy)
  - Partially Supported
  - Not supported: NotImplemented, Ellipsis, numbers.Complex, bytes, byte arrays, sets, frozen sets, generators, coroutines, async generators, modules, I/O objects, internal objects, slice objects ( though slicing is supported), classmethod
* - [3.3. Special method names](https://docs.python.org/3/reference/datamodel.html#special-method-names)
  - Supported
  -
* - [3.3.1. Basic customization](https://docs.python.org/3/reference/datamodel.html#basic-customization)
  - Partially Supported
  - Not supported: ``__new__`` , ``__del__`` , ``__bytes__`` , ``__format__`` , ``__hash__`` ,
* - [3.3.2. Customizing attribute access](https://docs.python.org/3/reference/datamodel.html#customizing-attribute-access)
  - Not Supported
  -
* - [3.3.2.1. Customizing module attribute access](https://docs.python.org/3/reference/datamodel.html#customizing-module-attribute-access)
  - Not Supported
  -
* - [3.3.2.2. Implementing Descriptors](https://docs.python.org/3/reference/datamodel.html#implementing-descriptors)
  - Not Supported
  -
* - [3.3.2.3. Invoking Descriptors](https://docs.python.org/3/reference/datamodel.html#invoking-descriptors)
  - Not Supported
  -
* - [3.3.2.4. __slots__](https://docs.python.org/3/reference/datamodel.html#slots)
  - Not Supported
  -
* - [3.3.2.4.1. Notes on using __slots__](https://docs.python.org/3/reference/datamodel.html#notes-on-using-slots)
  - Not Supported
  -
* - [3.3.3. Customizing class creation](https://docs.python.org/3/reference/datamodel.html#customizing-class-creation)
  - Not Supported
  -
* - [3.3.3.1. Metaclasses](https://docs.python.org/3/reference/datamodel.html#metaclasses)
  - Not Supported
  -
* - [3.3.3.2. Resolving MRO entries](https://docs.python.org/3/reference/datamodel.html#resolving-mro-entries)
  - Not Supported
  - [`super()`` is not supported
* - [3.3.3.3. Determining the appropriate metaclass](https://docs.python.org/3/reference/datamodel.html#determining-the-appropriate-metaclass)
  - Not relevant
  -
* - [3.3.3.4. Preparing the class namespace](https://docs.python.org/3/reference/datamodel.html#preparing-the-class-namespace)
  - Not relevant
  -
* - [3.3.3.5. Executing the class body](https://docs.python.org/3/reference/datamodel.html#executing-the-class-body)
  - Not relevant
  -
* - [3.3.3.6. Creating the class object](https://docs.python.org/3/reference/datamodel.html#creating-the-class-object)
  - Not relevant
  -
* - [3.3.3.7. Uses for metaclasses](https://docs.python.org/3/reference/datamodel.html#uses-for-metaclasses)
  - Not relevant
  -
* - [3.3.4. Customizing instance and subclass checks](https://docs.python.org/3/reference/datamodel.html#customizing-instance-and-subclass-checks)
  - Not Supported
  -
* - [3.3.5. Emulating generic types](https://docs.python.org/3/reference/datamodel.html#emulating-generic-types)
  - Not Supported
  -
* - [3.3.6. Emulating callable objects](https://docs.python.org/3/reference/datamodel.html#emulating-callable-objects)
  - Supported
  -
* - [3.3.7. Emulating container types](https://docs.python.org/3/reference/datamodel.html#emulating-container-types)
  - Partially Supported
  - Some magic methods not supported (e.g. ``__iter__`` )
* - [3.3.8. Emulating numeric types](https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types)
  - Partially Supported
  - Magic methods with swapped operands not supported (``__r*__``)
* - [3.3.9. With Statement Context Managers](https://docs.python.org/3/reference/datamodel.html#with-statement-context-managers)
  - Not Supported
  -
* - [3.3.10. Special method lookup](https://docs.python.org/3/reference/datamodel.html#special-method-lookup)
  - Not relevant
  -
* - [3.4. Coroutines](https://docs.python.org/3/reference/datamodel.html#coroutines)
  - Not Supported
  -
* - [3.4.1. Awaitable Objects](https://docs.python.org/3/reference/datamodel.html#awaitable-objects)
  - Not Supported
  -
* - [3.4.2. Coroutine Objects](https://docs.python.org/3/reference/datamodel.html#coroutine-objects)
  - Not Supported
  -
* - [3.4.3. Asynchronous Iterators](https://docs.python.org/3/reference/datamodel.html#asynchronous-iterators)
  - Not Supported
  -
* - [3.4.4. Asynchronous Context Managers](https://docs.python.org/3/reference/datamodel.html#asynchronous-context-managers)
  - Not Supported
  -
* - [4. Execution model](https://docs.python.org/3/reference/executionmodel.html#)
  - Not Relevant
  -
* - [4.1. Structure of a program](https://docs.python.org/3/reference/executionmodel.html#structure-of-a-program)
  - Not Relevant
  -
* - [4.2. Naming and binding](https://docs.python.org/3/reference/executionmodel.html#naming-and-binding)
  - Not Relevant
  - Names are bound at compile time in TorchScript
* - [4.2.1. Binding of names](https://docs.python.org/3/reference/executionmodel.html#binding-of-names)
  - Not Relevant
  - See ``global`` and ``nonlocal`` statements section
* - [4.2.2. Resolution of names](https://docs.python.org/3/reference/executionmodel.html#resolution-of-names)
  - Not Relevant
  - See ``global`` and ``nonlocal`` statements section
* - [4.2.3. Builtins and restricted execution](https://docs.python.org/3/reference/executionmodel.html#builtins-and-restricted-execution)
  - Not Relevant
  -
* - [4.2.4. Interaction with dynamic features](https://docs.python.org/3/reference/executionmodel.html#interaction-with-dynamic-features)
  - Not Supported
  - Python values cannot be captured
* - [4.3. Exceptions](https://docs.python.org/3/reference/executionmodel.html#exceptions)
  - Partially Supported
  - See ``try`` and ``raise`` statement section
* - [5. The import system](https://docs.python.org/3/reference/import.html)
  - Not Relevant
  -
* - [6. Expressions](https://docs.python.org/3/reference/expressions.html#)
  - Not Relevant
  - See expressions section
* - [6.1. Arithmetic conversions](https://docs.python.org/3/reference/expressions.html#arithmetic-conversions)
  - Supported
  -
* - [6.2. Atoms](https://docs.python.org/3/reference/expressions.html#atoms)
  - Not Relevant
  -
* - [6.2.1. Identifiers (Names)](https://docs.python.org/3/reference/expressions.html#atom-identifiers)
  - Supported
  -
* - [6.2.2. Literals](https://docs.python.org/3/reference/expressions.html#literals)
  - Partially Supported
  - [`bytesliteral``\ , ``imagnumber`` not supported
* - [6.2.3. Parenthesized forms](https://docs.python.org/3/reference/expressions.html#parenthesized-forms)
  - Supported
  -
* - [6.2.4. Displays for lists, sets and dictionaries](https://docs.python.org/3/reference/expressions.html#displays-for-lists-sets-and-dictionaries)
  - Partially Supported
  - Not supported: comprehension ifs, async iterators
* - [6.2.5. List displays](https://docs.python.org/3/reference/expressions.html#list-displays)
  - Supported
  -
* - [6.2.6. Set displays](https://docs.python.org/3/reference/expressions.html#set-displays)
  - Not Supported
  -
* - [6.2.7. Dictionary displays](https://docs.python.org/3/reference/expressions.html#dictionary-displays)
  - Supported
  - dict() constructor with kwargs doesn't work, dict comprehensions, dictionary unpacking
* - [6.2.8. Generator expressions](https://docs.python.org/3/reference/expressions.html#generator-expressions)
  - Not Supported
  -
* - [6.2.9. Yield expressions](https://docs.python.org/3/reference/expressions.html#yield-expressions)
  - Not Supported
  -
* - [6.2.9.1. Generator-iterator methods](https://docs.python.org/3/reference/expressions.html#generator-iterator-methods)
  - Not Supported
  -
* - [6.2.9.2. Examples](https://docs.python.org/3/reference/expressions.html#examples)
  - Not Supported
  -
* - [6.2.9.3. Asynchronous generator functions](https://docs.python.org/3/reference/expressions.html#asynchronous-generator-functions)
  - Not Supported
  -
* - [6.2.9.4. Asynchronous generator-iterator methods](https://docs.python.org/3/reference/expressions.html#asynchronous-generator-iterator-methods)
  - Not Supported
  -
* - [6.3. Primaries](https://docs.python.org/3/reference/expressions.html#primaries)
  - Supported
  -
* - [6.3.1. Attribute references](https://docs.python.org/3/reference/expressions.html#attribute-references)
  - Supported
  -
* - [6.3.2. Subscriptions](https://docs.python.org/3/reference/expressions.html#subscriptions)
  - Supported
  -
* - [6.3.3. Slicings](https://docs.python.org/3/reference/expressions.html#slicings)
  - Partially Supported
  - Tuple slicing with stride is not supported
* - [6.3.4. Calls](https://docs.python.org/3/reference/expressions.html#calls)
  - Partially Supported
  - Args unpack / kwargs unpack is not supported
* - [6.4. Await expression](https://docs.python.org/3/reference/expressions.html#await-expression)
  - Not Supported
  -
* - [6.5. The power operator](https://docs.python.org/3/reference/expressions.html#the-power-operator)
  - Supported
  -
* - [6.6. Unary arithmetic and bitwise operations](https://docs.python.org/3/reference/expressions.html#unary-arithmetic-and-bitwise-operations)
  - Partially Supported
  - Some bitwise operators are not implemented for primitive types (e.g. ``~x`` where ``x`` is an ``int`` is not currently supported)
* - [6.7. Binary arithmetic operations](https://docs.python.org/3/reference/expressions.html#binary-arithmetic-operations)
  - Partially Supported
  - See delimiters section
* - [6.8. Shifting operations](https://docs.python.org/3/reference/expressions.html#shifting-operations)
  - Not Supported
  -
* - [6.9. Binary bitwise operations](https://docs.python.org/3/reference/expressions.html#binary-bitwise-operations)
  - Supported
  -
* - [6.10. Comparisons](https://docs.python.org/3/reference/expressions.html#comparisons)
  - Supported
  -
* - [6.10.1. Value comparisons](https://docs.python.org/3/reference/expressions.html#value-comparisons)
  - Partially Supported
  - Dictionary equality checks are not currently supported
* - [6.10.2. Membership test operations](https://docs.python.org/3/reference/expressions.html#membership-test-operations)
  - Partially Supported
  - Not supported for TorchScript classes
* - [6.10.3. Identity comparisons](https://docs.python.org/3/reference/expressions.html#is-not)
  - Supported
  -
* - [6.11. Boolean operations](https://docs.python.org/3/reference/expressions.html#boolean-operations)
  - Supported
  -
* - [6.12. Conditional expressions](https://docs.python.org/3/reference/expressions.html#conditional-expressions)
  - Supported
  -
* - [6.13. Lambdas](https://docs.python.org/3/reference/expressions.html#lambda)
  - Not Supported
  -
* - [6.14. Expression lists](https://docs.python.org/3/reference/expressions.html#expression-lists)
  - Partially Supported
  - Iterable unpacking not supported
* - [6.15. Evaluation order](https://docs.python.org/3/reference/expressions.html#evaluation-order)
  - Supported
  -
* - [6.16. Operator precedence](https://docs.python.org/3/reference/expressions.html#operator-precedence)
  - Supported
  -
* - [7. Simple statements](https://docs.python.org/3/reference/simple_stmts.html#)
  - Supported
  -
* - [7.1. Expression statements](https://docs.python.org/3/reference/simple_stmts.html#expression-statements)
  - Supported
  -
* - [7.2. Assignment statements](https://docs.python.org/3/reference/simple_stmts.html#assignment-statements)
  - Supported
  -
* - [7.2.1. Augmented assignment statements](https://docs.python.org/3/reference/simple_stmts.html#augmented-assignment-statements)
  - Partially Supported
  - See delimiters section
* - [7.2.2. Annotated assignment statements](https://docs.python.org/3/reference/simple_stmts.html#annotated-assignment-statements)
  - Supported
  -
* - [7.3. The assert statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement)
  - Partially Supported
  - Exception message is not customizable
* - [7.4. The pass statement](https://docs.python.org/3/reference/simple_stmts.html#the-pass-statement)
  - Supported
  -
* - [7.5. The del statement](https://docs.python.org/3/reference/simple_stmts.html#the-del-statement)
  - Not Supported
  -
* - [7.6. The return statement](https://docs.python.org/3/reference/simple_stmts.html#the-return-statement)
  - Supported
  - Some other features of returning (e.g. behavior with try..finally) are unsupported
* - [7.7. The yield statement](https://docs.python.org/3/reference/simple_stmts.html#the-yield-statement)
  - Not Supported
  -
* - [7.8. The raise statement](https://docs.python.org/3/reference/simple_stmts.html#the-raise-statement)
  - Partially Supported
  - Exception message is not customizable
* - [7.9. The break statement](https://docs.python.org/3/reference/simple_stmts.html#the-break-statement)
  - Supported
  - Some other features of returning (e.g. behavior with try..finally) are unsupported
* - [7.10. The continue statement](https://docs.python.org/3/reference/simple_stmts.html#the-continue-statement)
  - Supported
  - Some other features of returning (e.g. behavior with try..finally) are unsupported
* - [7.11. The import statement](https://docs.python.org/3/reference/simple_stmts.html#the-import-statement)
  - Not Supported
  -
* - [7.11.1. Future statements](https://docs.python.org/3/reference/simple_stmts.html#future-statements)
  - Not Supported
  -
* - [7.12. The global statement](https://docs.python.org/3/reference/simple_stmts.html#the-global-statement)
  - Not Supported
  -
* - [7.13. The nonlocal statement](https://docs.python.org/3/reference/simple_stmts.html#the-nonlocal-statement)
  - Not Supported
  -
* - [8. Compound statements](https://docs.python.org/3/reference/compound_stmts.html#)
  - Irrelevant
  -
* - [8.1. The if statement](https://docs.python.org/3/reference/compound_stmts.html#the-if-statement)
  - Supported
  -
* - [8.2. The while statement](https://docs.python.org/3/reference/compound_stmts.html#the-while-statement)
  - Partially Supported
  - while..else is not supported
* - [8.3. The for statement](https://docs.python.org/3/reference/compound_stmts.html#the-for-statement)
  - Partially Supported
  - for..else is not supported
* - [8.4. The try statement](https://docs.python.org/3/reference/compound_stmts.html#the-try-statement)
  - Not Supported
  -
* - [8.5. The with statement](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement)
  - Partially Supported
  - [`__exit__`` is always called with ``exc_type``, ``exc_value``, and ``traceback`` set to None, even if an exception was raised, and ``__exit__``'s return value is ignored.
* - [8.6. Function definitions](https://docs.python.org/3/reference/compound_stmts.html#function-definitions)
  - Not Supported
  -
* - [8.7. Class definitions](https://docs.python.org/3/reference/compound_stmts.html#class-definitions)
  - Not Supported
  -
* - [8.8. Coroutines](https://docs.python.org/3/reference/compound_stmts.html#coroutines)
  - Not Supported
  -
* - [8.8.1. Coroutine function definition](https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition)
  - Not Supported
  -
* - [8.8.2. The async for statement](https://docs.python.org/3/reference/compound_stmts.html#the-async-for-statement)
  - Not Supported
  -
* - [8.8.3. The async with statement](https://docs.python.org/3/reference/compound_stmts.html#the-async-with-statement)
  - Not Supported
  -
* - [9. Top-level components](https://docs.python.org/3/reference/toplevel_components.html#)
  - Not Relevant
  -
* - [9.1. Complete Python programs](https://docs.python.org/3/reference/toplevel_components.html#complete-python-programs)
  - Not Relevant
  -
* - [9.2. File input](https://docs.python.org/3/reference/toplevel_components.html#file-input)
  - Not Relevant
  -
* - [9.3. Interactive input](https://docs.python.org/3/reference/toplevel_components.html#interactive-input)
  - Not Relevant
  -
* - [9.4. Expression input](https://docs.python.org/3/reference/toplevel_components.html#expression-input)
  - Not Relevant
  -
```
