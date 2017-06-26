$(document).ready(function(){
    printJava("Hello World!")
});

function printJava(String msg){
    JavaBridge.print(msg);
}