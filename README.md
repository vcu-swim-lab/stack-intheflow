# StackInTheFlow <img src="https://github.com/vcu-swim-lab/stack-intheflow/blob/master/src/main/resources/icons/main.png" width="65" height="60">

An IntelliJ plugin to query StackOverflow.

### Search StackOverflow Without Leaving the IDE
StackInTheFlow integrates seamlessly with Intellij to allow you to find the solutions to the development problems you face without ever leaving the *flow* of your development environment.

![Search](https://i.imgur.com/Rt5tYun.gif)

### Automatically Generate Queries from your Code
Take your automation one step further by having StackInTheFlow generate queries for you. Either generate a query from an editor tab or select a block of code, then right-click and select Auto Query!

![Auto](https://i.imgur.com/yB7fH5N.gif)

### Receive Smart Recommendations to Accelerate your Workflow
If enabled, StackInTheFlow with make suggestions of StackOverflow articles when it thinks they will help you out with your current task, ensuring relevant information is always at your fingertips. Never manually search for the meaning behind a cryptic error message again!

![Error](https://i.imgur.com/L9WO8OF.png)

### Sort & Filter
Refine your searches by sorting by different metrics and filtering by specific tags. If you never want to leave the comfort of your keyboard, you can even press `<TAB>` to add the previous word as a tag.

![Filter](https://i.imgur.com/NuQF2cl.gif)

********

### Running in Android Studio
In order to run StackInTheFlow within Android Studio you must set the `STUDIO_JDK` environment variable to point to an Oracle JDK installation on your machine. You can read about setting Android Studio environment variables [here](https://developer.android.com/studio/command-line/variables.html#set).  
For example on Windows in the command prompt you would run:
```
set STUDIO_JDK="C:\Program Files\Java\jdk_version_something"
```
Alternatively you can:
 * Search for System (Control Panel)
 * Click **Advanced Systems Settings**
 * In **System Variables** select **New**
 * Set `STUDIO_JDK` to point to where the JDK software is located, for example, C:\Program Files\Java\jdk1.8.0_144.

## Contributors
Chase Greco ([Zerthick](https://github.com/zerthick)) - Project Lead  

Chris Carpenter ([carpentercr](https://github.com/carpentercr)) - Logging  
John Coogle ([cooglejj](https://github.com/cooglejj)) - Core  
Jeet Gajjar ([kakorrhaphio](https://github.com/kakorrhaphio)) - Core  
Tyler Haden ([tylerjohnhaden](https://github.com/tylerjohnhaden)) - Core  
Kevin Ngo ([cptcomo](https://github.com/cptcomo)) - UI  

Kosta Damevski  ([damevski](https://damevski.github.io)) - Project Advisor  

*This plugin is produced by the [SWIM Lab](http://vcu-swim-lab.github.io/) @ Virginia Commonwealth University*

## Install
[StackInTheFlow](https://plugins.jetbrains.com/plugin/9653-stackintheflow)

## License
[MIT](./LICENSE)
