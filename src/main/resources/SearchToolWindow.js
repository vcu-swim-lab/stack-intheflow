
/* Variables and Objects for the entire UI */

var questionsList; //An array of Question objects that hold the information for the question results
var questionSections; //An array of <section> tags in HTML, what the user sees
var numQuestions; //Number of results
var charCutoff; //The ideal excerpt cutoff - number of characters
var searchTags; //SearchTag object, responsible for search tags logic
var uiSettings; //UISettings object, responsible for the UI color schemes
var searchMethod; //A string representing the current selected sort type (Relevance, Votes, New, Active)
var queryHistory; //History object, holds the query history
var lastQueryType; //always has the most recent query type, used for log correlation


//This function replaces $(document).ready(), which for some reason does not get called if a second instance of the UI is opened. See Pull Req. #107
//initialize() is called in Java createScene() when the JavaBridge gets set up.
function initialize(){
    charCutoff = 300; //Change this to make excerpts longer/shorter

    searchTags = new SearchTags();
    uiSettings = new UISettings();
    queryHistory = new History();
    lastQueryType = "-1";

    searchMethod = "RELEVANCE";

    /*Set up the event listeners for elements that are NOT part of the question results. Those event listeners are added in generateListeners().
        Diff between ".on" and ".click()":
            ".on" with the event as a parameter means that the event listener is added to the parent, and is triggered when a child element is triggered with the event
            ".click()" or other events are applied to the element itself. This means that if the element is destroyed, the listener is destroyed
    */

    //Limit the width of the history window, see issue #116
    limitDropdownWidth(Math.round($(window).width() * 0.8));
    $(window).resize(function(){
       limitDropdownWidth(Math.round($(window).width() * 0.8));
    });

    $('#searchBox').keydown(function(e){
        if(e.keyCode == 9 && !e.shiftKey){ //Tab was pressed, make the last word in the search box a search tag
            e.preventDefault(); //Don't do what a tab usually does (advance focus to next element)
            var words = $('#searchBox').val().split(" ").filter((item) => item != ''); //Convert search box value to an array of strings, remove extraneous spaces

            if(words.length == 0){ //Don't add a tag if there is nothing in the box
                return;
            }

            searchTags.add(words[words.length - 1]);
            words.pop(); //Remove last element
            words = words.toString().split(",").join(" "); //pop() returns comma separated tags, so replace commas with spaces

            setSearchBox(words);

            search(true);
        }
        hideAutoQueryIcon();
    });

    $(document).on('keypress', '#searchBox', function(e){
        if(e.which == 13){ //The enter key
            search(true);
        }
    });

    $("#searchButton").click(function(){
        search(true);
    });

    $("#searchTags").on("click", "li", function(e){ //Click on a search tag in the bar to remove it
        searchTags.remove($(this).html());
        search(true);
    });

    $("#searchMethodsMenu").on("click", "li", function(e){ //A new sort type is selected
        $('#searchMethodsMenu').find(".selectedItem").removeClass("selectedItem"); //Remove styling for the current sort type
        $(this).children().first().addClass("selectedItem"); //And add the styling to this
        var dropdownVal = $(this).children().first().html().toLowerCase();

        //Convert UI options to API
        if(dropdownVal == "relevance")
            dropdownVal = "RELEVANCE";
        else if(dropdownVal == "votes")
            dropdownVal = "VOTES";
        else if(dropdownVal == "newest")
            dropdownVal = "CREATION";
        else if(dropdownVal == "active")
            dropdownVal = "ACTIVITY";

        searchMethod = dropdownVal;
        search(true);
    });

    $('#historyButton').click(function(){ //History button event listener
        queryHistory.updateUI();
    });

    $('#historyMenu').on('click', 'li', function(e){ //History dropdown element listener, - requery the item clicked
        //Get the query and tags of the element clicked
        var index = $(this).index();
        var query = queryHistory.getQuery(index);
        var tags = queryHistory.getTag(index).split(" ").filter((item) => item != '');

        setSearchBox(query);
        searchTags.setTags(tags);

        search(false);
    });

    $('#settingsButton').click(function(){ //Settings button event listener
        JavaBridge.openSettings();
    });

    //Activate Tooltips
    $('[data-toggle="tooltip"]').tooltip({
        trigger: 'hover'
    });

    $('[data-toggle="tooltip"]').on('click', function(){
        $(this).tooltip('hide');
    });
}

/*
    Object that handles the UI color theme. Changes between the light and dark theme
*/
function UISettings(){
    this.isDark = false;

    this.updateUI = function(){
        //Enable and disable the 2 different spreadsheets as appropriate.
        if(this.isDark){
            document.getElementById("defaultSheet").disabled = true;
            document.getElementById("darkSheet").disabled = false;
        }
        else {
            document.getElementById("defaultSheet").disabled = false;
            document.getElementById("darkSheet").disabled = true;
        }
    }
}

// This function is called by JavaBridge in Java, which tells the UI to update the color theme.
function updateUISettings(isDark){
    uiSettings.isDark = isDark;
    uiSettings.updateUI();
}

/*
    Object that handles the History feature of the UI
*/
function History(){
    //Arrays to hold queries and tags, index links the queries and the tags.
    this.queries = new Array();
    this.tags = new Array();

    this.add = function(query, tag){
        this.queries.push(query);
        this.tags.push(tag);
    }

    //For use case, index of 0 is to be the latest query, but that is the last element in the array. Getters and setters reverse order

    this.getQuery = function(index){
        return this.queries[this.queries.length - index - 1];
    }

    this.getTag = function(index){
        return this.tags[this.tags.length - index - 1];
    }

    this.setQuery = function(index, query){
        this.queries[this.queries.length - index - 1] = query;
    }

    this.setTag = function(index, tag){
        this.tags[this.tags.length - index - 1] = tag;
    }

    this.getMostRecentQuery = function(){
        return this.getQuery(0);
    }

    this.getMostRecentTag = function(){
        return this.getTag(0);
    }

    this.updateUI = function(){
        $('#historyMenu').empty(); //To update, clear out all the history and regenerate
        if(this.queries.length > 0){ //There is at least one query already made
            for(var i = this.queries.length - 1; i >= 0; i--){
                /*
                    History HTML is of the form:
                    <ul ... historyMenu>
                        <li>
                            <span>Query string here [tag1 tag2 tag3]</span>
                        </li>
                        ...
                    </ul>
                */
                var li = $("<li>");
                var span = $("<span>").html(this.queries[i] + (this.tags[i] == "" ? "" : " [" + this.tags[i] + "]"));
                $(li).append(span);
                $('#historyMenu').append(li);
            }
        }
        else {
            var message = $("<span>").html("No history to show yet...");
            $('#historyMenu').append(message);
        }
    }
}

/*
    Object to handle the tags to search feature
*/
function SearchTags(){
    this.tags = new Array();

    this.add = function(newTag){
        if(!this.contains(newTag)){
            this.tags.push(newTag);
            JavaBridge.updatePersonalSearchModel(newTag, 2);
            this.updateUI();
        }
    }

    this.remove = function(tagToRemove){
        for(var i = 0; i < this.tags.length; i++){
            if(this.tags[i] === tagToRemove){
                this.tags.splice(i, 1);
            }
        }
        this.updateUI();
    }

    this.contains = function(tagToCheck){
        for(var i = 0; i < this.tags.length; i++){
            if(this.tags[i] == tagToCheck){
                return true;
            }
        }
        return false;
    }

    this.setTags = function(tags){ //tags is an array of Strings
        this.tags = tags;
        this.updateUI();
    }

    this.updateUI = function(){
        $('#searchTags').empty();
        for(var i = 0; i < this.tags.length; i++){
            var tag = $('<li>').html(this.tags[i]);
            $('#searchTags').append(tag);
        }

        // add/remove spacing in the UI to accommodate for the presence of search tags
        if(this.tags.length > 0){
            $('#content').removeClass("contentStart").addClass("contentWithSearchTags");
        }
        else {
            $('#content').removeClass("contentWithSearchTags").addClass("contentStart");
        }
     }

    this.clear = function(){
        this.tags = new Array();
        this.updateUI();
    }

    this.getQuerySyntax = function(){ //Returns the syntax for the API advanced syntax. Ex. [tag1] [tag2] [tag3]
        var query = "";

        for(var i = 0; i < this.tags.length; i++){
            query += "[" + this.tags[i] + "]" + (i == this.tags.length - 1 ? "" : " ");
        }

        return query;
    }

    this.toString = function(){ //Return a string representation of the tags separated by spaces
        var string = "";
        for(var i = 0; i < this.tags.length; i++){
            string += this.tags[i] + (i == this.tags.length - 1 ? "" : " ");
        }
        return string;
    }
}

//This function is called in Java autoQuery(...) which is triggered by the backend Components
//Reasoning is either "action" or "difficulty"
function autoSearch(query, backoff, reasoning){
    reset();
    searchTags.clear();

    if(query == ""){
        var message = $("<h2>").html("Unable to generate query, not enough data points.");
        $('#questions').append(message);
        hideAutoQueryIcon();
        return;
    }

    tags = ""; //Wipe out tags for autoqueries
    JavaBridge.autoQuery(query, tags, backoff, true, reasoning); //Call JavaBridge to execute the actual query.

    showAutoQueryIcon(reasoning);
}

//This function is called when the user is actively triggering the search manually.
//(Clicking the search button, pressing enter, adding/removing tags, clicking on history query, clicking on another sort type, etc.)
function search(addToHistory){
    var query = $('#searchBox').val().split("<script>").join("").split("</script>").join("");

    while(query.includes("<script>") || query.includes("</script>")){
        query = query.split("<script>").join("").split("</script>").join("");
    }

    if(addToHistory){
        if(queryHistory.getMostRecentQuery() == query){ //Don't add a new entry for the history if the user is just modifying the latest query (Ex. adding/removing tags)
            queryHistory.setTag(0, searchTags.toString()); //Just update the tags for the latest entry
            addToHistory = false;
        }
    }

    var tags = searchTags.getQuerySyntax();

    if(query == "" && tags == ''){ //Don't make empty queries
        return;
    }

    reset();

    JavaBridge.searchButtonClicked(query, tags, searchMethod, addToHistory); //Call JavaBridge to execute the actual query
    hideAutoQueryIcon();
}

function resetSearchTags(){
    searchTags.clear();
}

function showAutoQueryIcon(reasoning){
    var message = "";
    if(reasoning == "action"){
        message = "This query was automatically generated by user request";
    } else if(reasoning == "difficulty"){
        message = "This query was automatically generated by difficulty detection";
    } else if(reasoning == "compiler"){
        message = "This query was automatically generated due to a compiler error";
    } else if(reasoning == "runtime"){
        message = "This query was automatically generated due to a runtime error";
    }

    //This line is necessary in order to change the text
    $("#autoQueryIcon").attr("title", message).tooltip('fixTitle');

    $("#autoQueryIcon").removeClass("hidden");
    $("#searchBox").addClass('removeBorderLeft'); //In the default state, the search box has a left border. Remove this if the icon is there.
}

function hideAutoQueryIcon(){
    $("#searchBox").removeClass('removeBorderLeft'); //In the default state, the search box has a left border. Re-add this if the icon is being removed.
    $("#autoQueryIcon").addClass("hidden");
}

//Currently, this function is being called in Java autoQuery and errorQuery. This is because the auto queries are relevance sorted.
//This updates the UI to reflect that.
function updateUISearchType(searchType){
    $('#searchMethodsMenu').find(".selectedItem").removeClass("selectedItem");
    var children = $('#searchMethodsMenu').children();
    $(children).each(function(){
        if($(this).children().first().html().toLowerCase() == searchType.toLowerCase()){
            $(this).children().first().addClass("selectedItem");
        }
    });
    searchMethod = searchType.toUpperCase();
}

function setSearchBox(newValue){
    $('#searchBox').val(newValue);
}

function logQuery(queryType){
    query = $("#searchBox").val();
    tags = searchTags.toString();
    var tagArray = tags.split(" ");
    var tagLength = tagArray.length;
    for (var i = 0; i < tagLength; i++){
        tagArray[i] = hashCode(tagArray[i]);
    }
    var tagString = tagArray.toString();

    var message = '"QueryEventType":' + '"' + queryType + '"' + ', ' + '"Tags":[' +
                    tagString.split(" ").join(", ") + '], ' + '"sort":' + '"' + searchMethod.toLowerCase() + '"' + '}';

    JavaBridge.log(message);
    lastQueryType = queryType;
}

function logResultEvent(resultType, index){
    var totalResults = numQuestions;

    var message = '"ResultEventType":' + '"' + resultType + '"' + ', ' + '"SourceQuery":"' + lastQueryType + '", ' + '"Index":' + index + ', ' + '"Total":' + totalResults + '}';

    JavaBridge.log(message);
}

//This function is called from Java to ensure that the searchBox has the latest query.
function addCurrentQueryToHistory(){
    queryHistory.add($('#searchBox').val(), searchTags.toString());
}

function reset(){
    setSearchBox('');
    $('#questions').empty();
    questionsList = new Array();
    questionSections = new Array();
    numQuestions = 0;
    $('html, body').animate({ scrollTop: 0 }, 'fast'); //Scroll back to the top
}

/* Below is code for the search results themselves */

/*
    Object that represents a Question result
*/
function Question(title, body, tags, link){
    this.title = title;
    this.body = body;
    this.tags = tags;
    this.link = link;
    this.codeTags = new Array(); //An array of CodeTag

    this.findCodeTags = function(){ //Find the start and end index of code blocks
        var lastStartIndex = -1;
        var lastEndIndex = -1;
        while(true){
            var start = this.body.indexOf("<pre>", lastStartIndex + 1);
            var end = this.body.indexOf("</pre>", lastEndIndex + 1);

            if(start != end && start != -1){
                this.codeTags.push(new CodeTag(start, end));
                lastStartIndex = start;
                lastEndIndex = end;
            }
            else {
                break;
            }
        }
    }

    this.hasCodeTags = function(){
        return this.codeTags.length > 0;
    }

    this.lastCodeOpenTag = function(){
        if(this.hasCodeTags()){
             return this.codeTags[this.codeTags.length - 1].open;
        }
        else {
            return -1;
        }
    }

    this.lastCodeCloseTag = function(){
        if(this.hasCodeTags()){
            return this.codeTags[this.codeTags.length - 1].close;
        }
        else {
            return -1;
        }
    }

    /*
        This function returns the final number of characters to include in the excerpt. It tries to get as close to charCutoff as possible,
        but it has to take into account a couple of factors:

        Sometimes charCutoff happens to be in the middle of a code block. Cutting the content off here is awkward, and it also breaks the formatting of the code blocks
        So this will try to either place the cutoff before the start of a code block or after the end of a code block, but within the tolerance
    */
    this.getCutoff = function(){
        var tolerance = 100; //Modify this to change how much content is willing to be added on or removed in order to meet criteria above.
        for(var i = this.codeTags.length - 1; i >= 0; i--){
            if(Math.abs(this.codeTags[i].close - charCutoff) < tolerance){ //First, try to include the entire code block if it is within tolerance
                return this.codeTags[i].close;
            }
            else if(Math.abs(this.codeTags[i].open - charCutoff) < tolerance){ //Then, if the start of a code block is within tolerance, cut off before
                return this.codeTags[i].open - 1;
            }
            else if(this.codeTags[i].open < charCutoff && this.codeTags[i].close > charCutoff){ //The code block is too large but charCutoff is in the block. Safest is to just cutoff before.
                return this.codeTags[i].open - 1;
            }
        }

        return charCutoff; //If it gets to this point, then just return the default cutoff.
    }

    this.getShortenedContent = function(){
        return this.body.substring(0, this.getCutoff());
    }

    this.hasMoreContent = function(){
        return this.body.length > this.getShortenedContent().length;
    }
}

/*
    Object that represents a code block. Simply for the purpose of easiness to code/read - in place of an array.
*/
function CodeTag(open, close){
    this.open = open;
    this.close = close;
}

//This function is called in Java updateQuestionList(...)
//This function receives each of the question results one by one.
//JavaBridge doesn't support passing Objects (I don't think), so Questions are decomposed in Java and rebuilt into a Question object in JS.
// title, body, link - String, tags - Array of Strings
function getQuestion(title, body, tags, link){
    tagsString = new Array();
    for(i = 0; i < tags.length; i++){
        tagsString.push(tags[i].toString());
    }
    questionsList.push(new Question(title, body, tagsString, link));;
    questionsList[numQuestions].findCodeTags();
    numQuestions++;
}

/*
    Takes the questionList, which holds the Question objects, and creates the HTML elements (and updates questionSections in the process)
*/
function displayQuestions(){
    if(numQuestions > 0){
        for(var i = 0; i < numQuestions; i++){
            appendNewResultSkeleton(i); //Generate the skeleton of a question result.
            var questionSection = questionSections[i];

            $(questionSection).find(".searchResultTitle").html(questionsList[i].title); //Set title

            var questionBody = $(questionSection).find(".questionBody");
            $(questionBody).html(questionsList[i].getShortenedContent()); //Set body to the excerpt

            if(questionsList[i].hasMoreContent()){ //If the content had to be cut off, add in the "more"
                var excerptController = $("<div>").addClass("excerptController").html("More");

                if($(questionBody).children().length > 0){
                    //This is logic for whether or not to place the "more" on a new line or at the end of the last line.
                    //Generally, in-line if the last element is text-based, new line if the cutoff is after code block.
                    var lastChild = $(questionBody).children().last();
                    if($(lastChild).is("PRE")){
                        excerptController.removeClass('inlineExcerptController');
                        excerptController.addClass('blockExcerptController');
                        $(questionBody).append(excerptController);
                    }
                    else {
                        excerptController.addClass('inlineExcerptController');
                        excerptController.removeClass('blockExcerptController');
                        $(lastChild).append(excerptController);
                    }
                }
                else {
                    //This fixes a bug where the post starts off with a really long code block. The cutoff is then before the code block, so the excerpt is empty.
                    //The controller was attempted to be added into the last child element, but there is no last child element because the excerpt is empty.
                    //So this makes a "fake" element to append the controller
                    var newChild = $("<span>");
                    excerptController.removeClass('inlineExcerptController');
                    excerptController.addClass('blockExcerptController');
                    $(newChild).append(excerptController);
                    $(questionBody).append(newChild);
                }
            }

            //Display the tags associated with the result
            var questionTagsContainer = $(questionSection).find(".questionTags");
            var unorderedList = $("<ul>");

            for(var j = 0; j < questionsList[i].tags.length; j++){
                var tagItem = $("<li>").html(questionsList[i].tags[j]);
                $(unorderedList).append(tagItem);
            }
            $(questionTagsContainer).append(unorderedList);
        }
    }
    else {
        var message = $("<h2>").text("Sorry, querying \"" + $('#searchBox').val() + "\" yielded no results. :(");
        $('#questions').append(message);
    }
    uiSettings.updateUI();
}

/*
    Generates the skeleton of a question result. The HTML skeleton is as follows:

    <section class="searchResultItem">
        <span class="hidden" id="...">...</span>
        <div class="searchResultItemBody">
            <div class="row">
                <div class="col-xs-12">
                    <h3 class="searchResultTitle">...</h3>
                    <div class="questionBody contentShortened">
                        ...
                    </div>
                </div>
            </div>
        </div>
    </section>
*/
function appendNewResultSkeleton(i){
    var questionSection = $("<section>").addClass("searchResultItem");
    var questionBodyDiv = $("<div>").addClass("searchResultItemBody");
    var rowDiv = $("<div>").addClass("row");
    var questionBodyContentContainer = $("<div>").addClass("col-xs-12");
    var questionTitle = $("<h3>").addClass("searchResultTitle");
    var questionBodyContent = $("<div>").addClass("questionBody contentShortened");

    $(questionBodyContentContainer).append(questionTitle);
    $(questionBodyContentContainer).append(questionBodyContent);

    $(rowDiv).append(questionBodyContentContainer);

    $(questionBodyDiv).append(rowDiv);

    var questionIndex = $('<span>').addClass('hidden').attr('id', 'questionIndex').html(i);
    $(questionSection).append(questionIndex);
    $(questionSection).append(questionBodyDiv);

    var rowDiv2 = $("<div>").addClass("row");
    var tagsDiv = $('<div>').addClass("questionTags col-xs-12");

    $(rowDiv2).append(tagsDiv);

    $(questionSection).append(rowDiv2);

    $(questions).append(questionSection);
    questionSections.push(questionSection);
}

/*
    Generates listeners necessary for question results
*/
function generateListeners(){
    for(var i = 0; i < questionSections.length; i++){
        // Expand/contract results
        $(questionSections[i]).on("click", ".excerptController", function(e){
            var clickedSection = $(this).closest('.searchResultItem');
            var index = $(clickedSection).find('#questionIndex').html();
            var questionBody = $(clickedSection).find('.questionBody');
            if($(this).html() == 'More'){
                $(questionBody).html(questionsList[index].body);
                $(this).html("Less");
                JavaBridge.updatePersonalSearchModel(questionsList[index].tags, 1);
                logResultEvent("expand", index);
            }
            else {
                $(questionBody).html(questionsList[index].getShortenedContent());
                if(questionBody.children().length == 0){
                    var newChild = $("<span>");
                    $(newChild).append($(this));
                    $(questionBody).append(newChild);
                }
                $(this).html("More");
                logResultEvent("contract", index);
            }

            var lastChild = $(questionBody).children().last();
            if($(lastChild).is("PRE")){
                $(this).removeClass('inlineExcerptController');
                $(this).addClass('blockExcerptController');
                $(questionBody).append($(this));
            }
            else {
                $(this).addClass('inlineExcerptController');
                $(this).removeClass('blockExcerptController');
                $(lastChild).append($(this));
            }
        });

        $(questionSections[i]).on("click", ".questionTags li", function(e){ //Click on question tags to add to search tags
            searchTags.add($(this).html());
            search(true);
        });

        $(questionSections[i]).on("click", ".searchResultTitle", function(e){ //Click on title to open in browser
            var clickedSection = $(this).closest('.searchResultItem');
            var index = $(clickedSection).find('#questionIndex').html();
            JavaBridge.updatePersonalSearchModel(questionsList[index].tags, 3);
            JavaBridge.openInBrowser(questionsList[index].link);
            logResultEvent("browser", index);
        });
    }

    $(document).on("click", "a", function(e){ //Intercept link clicks to open in browser. WebView breaks if links are opened in it.
        e.preventDefault();
        JavaBridge.openInBrowser($(this).attr('href'));
    });
}

/*
    Set the max width of the history dropdown elements according to screen size, add scrollbar if too long.
*/
function limitDropdownWidth(width){
    $('#historyMenu').css('max-width', width + "px");
}

function hashCode(string) {

  var hash = 0, length = string.length, i = 0;

  if ( length > 0 )

    while (i < length)

      hash = (hash << 5) - hash + string.charCodeAt(i++) | 0;

  return hash;
}
