
$(document).ready(function(){
    JavaBridge.print("Hello");
});

$("#searchButton").click(function(){
    JavaBridge.searchButtonClicked($('#searchBar').val());
});
