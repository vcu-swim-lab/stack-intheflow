
$(document).ready(function(){
    JavaBridge.print("Hello");
});

$("#searchButton").click(function(){
    $('#content').text($('#searchBar').val());

    JavaBridge.print("Click!");
});
