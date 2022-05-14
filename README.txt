In this project, we classified nearly 1,26,536 bugs reports of different open-source projects (Mozilla, NetBeans etc) into nine different categories 
using machine learning techniques. The bugs are classified based on their textual features (bug description).
We collected a sum of 1,26,537 bugs reports from the open source bug tracking
system: Bugzilla. We extracted all the bug reported in Bugzilla from January 5, 2015 to December 31, 2019 for all the projects. 
Based on the textual features of the description, the bugs are classified into nine categories. 
These classes are Clean, Concurrency, Crash, Network, Performance, Polish, Regression, Security, and Usability. 
The classes are considered based on relevant key words present in the description of the bug.

Code for this project written in Python.




Different Bugs labeling:


Regression Bugs: Regression bug arise whenever some previously existing functionality stops working and is no longer supported by the system.

Security Bugs: A security bug is a security vulnerability that allow some unintended user to have access of the system and thus may harm or 
damage the software or to the people using software. Security bugs are generally given priority and are hidden from other 
users when recognized for security reasons.

Polish Bugs: Polish bugs represent minor issues and fixes required in the system.

Cleanup Bugs: Cleanup Bugs also represent the minor fixes or recommendations in form of removing 
some feature or moving some feature from one module to another in a system.

Field Crash Bugs:  Bugs that trigger crash in different usages scenarios. 

Blocking Bugs: Blocking bugs are software bugs that prevent other bugs from being fixed. 
These blocking bugs may increase maintenance costs, reduce overall quality and delay the release of the software systems.

Performance Bugs: Bugs that effect on the performance or responsiveness of the application.
