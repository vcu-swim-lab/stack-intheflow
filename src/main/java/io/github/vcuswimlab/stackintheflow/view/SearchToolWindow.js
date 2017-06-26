function printJava(msg){
    JavaBridge.print(msg);
}

$("#searchButton").click(function(){
    console.log($('#searchBar').val());
    $('#content').text($('#searchBar').val());
});