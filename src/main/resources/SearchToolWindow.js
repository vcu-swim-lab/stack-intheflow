
var questionsList;
var numQuestions;
var charCutoff;
var questionSections;
var searchTags;
var uiSettings;
var searchMethod;
var queryHistory;

$(document).ready(function(){
    charCutoff = 300;
    searchTags = new SearchTags();
    uiSettings = new UISettings();
    queryHistory = new History();

    searchMethod = "RELEVANCE";

    $('#searchBox').keydown(function(e) {
        if(e.keyCode == 9 && !e.shiftKey) {
            e.preventDefault();

            var words = $('#searchBox').val().split(" ");
            searchTags.add(words[words.length - 1]);
            words.splice(words.length - 1, 1);
            $('#searchBox').val(words);
            search(true);
        }
        $("#autoQueryIcon").addClass("hidden");
    });

    $("#searchTags").on("click", "li", function(e){
        searchTags.remove($(this).html());
        search(true);
    });

    $("#searchMethodsMenu").on("click", "li", function(e){
        $('#searchMethodsMenu').find(".selectedItem").removeClass("selectedItem");
        $(this).children().first().addClass("selectedItem");
        var dropdownVal = $(this).children().first().html().toLowerCase();
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

    $('#historyButton').click(function(){
        JavaBridge.print("Click");
        queryHistory.updateUI();
    });

    $('#historyMenu').on('click', 'li', function(e){
        var index = $(this).index();
        var query = queryHistory.getQuery(index);
        var tags = queryHistory.getTag(index).split(" ").filter((item) => item != '');
        setSearchBox(query);
        searchTags.setTags(tags);

        search(false);
    });

    //Activate Tooltips
    $('[data-toggle="tooltip"]').tooltip();
});

function UISettings(){
    this.isDark = false;

    this.updateUI = function(){
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

function updateUISettings(isDark){
    uiSettings.isDark = isDark;
    uiSettings.updateUI();
}

function History(){
    this.queries = new Array();
    this.tags = new Array();

    this.add = function(query, tag){
        this.queries.push(query);
        this.tags.push(tag);
    }

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
        $('#historyMenu').empty();
        if(this.queries.length > 0){
            for(var i = this.queries.length - 1; i >= 0; i--){
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


function SearchTags(){
    this.tags = new Array();

    this.add = function(newTag){
        if(!this.contains(newTag)){
            this.tags.push(newTag);
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

    this.setTags = function(tags){
        this.tags = tags;
        this.updateUI();
    }

    this.updateUI = function(){
        $('#searchTags').empty();
        for(var i = 0; i < this.tags.length; i++){
            var tag = $('<li>').html(this.tags[i]);
            $('#searchTags').append(tag);
        }

        if(this.tags.length > 0){
            $('#content').removeClass("contentStart").addClass("contentWithSearchTags");
        }
        else {
            $('#content').removeClass("contentWithSearchTags").addClass("contentStart");
        }

        JavaBridge.debugBreakpoint();
     }

    this.clear = function(){
        this.tags = new Array();
        this.updateUI();
    }

    this.getQuerySyntax = function(){
        var query = "";

        for(var i = 0; i < this.tags.length; i++){
            query += "[" + this.tags[i] + "]" + (i == this.tags.length - 1 ? "" : " ");
        }

        return query;
    }

    this.toString = function(){
        var string = "";
        for(var i = 0; i < this.tags.length; i++){
            string += this.tags[i] + (i == this.tags.length - 1 ? "" : " ");
        }
        return string;
    }
}

//Reasoning is either "manual" or "difficulty"
function autoSearch(query, backoff, reasoning){
    reset();
    searchTags.clear();
    tags = "";
    JavaBridge.autoQuery(query, tags, backoff, true, reasoning);
    showAutoQueryIcon(reasoning);
}

function setSearchBox(query){
    $('#searchBox').val(query);
}

function search(addToHistory){
    if(addToHistory){
        if(queryHistory.getMostRecentQuery() == $('#searchBox').val()){
            queryHistory.setTag(0, searchTags.toString());
            addToHistory = false;
        }
    }
    reset();
    var query = $('#searchBox').val();
    var tags = searchTags.getQuerySyntax();
    JavaBridge.searchButtonClicked(query, tags, searchMethod, addToHistory);
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

    $("#autoQueryIcon").attr("title", message).tooltip('fixTitle');
    $("#autoQueryIcon").removeClass("hidden");
}

function hideAutoQueryIcon(){
    $("#autoQueryIcon").addClass("hidden");
}

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

function logQuery(queryType){
    query = $("#searchBox").val();
    tags = searchTags.toString();
    var message = "[query_event]<type>" + queryType + "</type><query>" + query + "</query><tags>" +
                    tags.split(" ").join(", ") + "</tags>" + "<sort>" + searchMethod.toLowerCase() + "</sort>";
    JavaBridge.log(message);
}

function logResultEvent(resultType, index){
    var totalResults = numQuestions;
    var message = "[result_event]<type>" + resultType + "</type><index>" + index + "</index><total>" + totalResults + "</total>";
    JavaBridge.log(message);
}

$("#searchButton").click(function(){
    search(true);
});

$(document).on('keypress', '#searchBox', function(e){
    if(e.which == 13){
        search(true);
    }
});

function addCurrentQueryToHistory(){
    queryHistory.add($('#searchBox').val(), searchTags.toString());
}

function reset(){
    $('#questions').empty();
    questionsList = new Array();
    questionSections = new Array();
    numQuestions = 0;
    $('html, body').animate({ scrollTop: 0 }, 'fast');
}

function Question(title, body, tags, link){
    this.title = title;
    this.body = body;
    this.tags = tags;
    this.link = link;
    this.codeTags = new Array();
    this.htmlTags = new Array();

    this.findCodeTags = function(){
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

    this.findHTMLTags = function(){
        var lastStartIndex = -1;
        var lastEndIndex = -1;
        while(true){
            var startMatch = this.body.regexIndexOf("<\\w[^>]*>", lastStartIndex + 1);
            var startIndex = parseInt(startMatch.index);
            var endMatch = this.body.regexIndexOf("<\\/[^>]+>", lastEndIndex + 1);
            var endIndex = parseInt(endMatch.index);

            if(startIndex != endIndex && startIndex != -1){
                this.htmlTags.push(new HTMLTag(startIndex, endIndex, endMatch.length));
                lastStartIndex = startIndex;
                lastEndIndex = endIndex;
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

    this.getCutoff = function(){
        var tolerance = 100;
        for(var i = this.codeTags.length - 1; i >= 0; i--){
            if(Math.abs(this.codeTags[i].close - charCutoff) < tolerance){
                return this.codeTags[i].close;
            }
            else if(Math.abs(this.codeTags[i].open - charCutoff) < tolerance){
                return this.codeTags[i].open - 1;
            }
            else if(this.codeTags[i].open < charCutoff && this.codeTags[i].close > charCutoff){
                return this.codeTags[i].open - 1;
            }
        }


        for(var i = 0; i < this.htmlTags.length; i++){
            if(charCutoff - this.htmlTags[i].open < this.htmlTags[i].length - 1 &&
                charCutoff - this.htmlTags[i].open > 0){
                return this.htmlTags[i].open - 1;
            }
            else if(charCutoff - this.htmlTags[i].close < this.htmlTags[i].length - 1 &&
                charCutoff - this.htmlTags[i].close > 0){
                return this.htmlTags[i].close + this.htmlTags[i].length;
            }
        }

        return charCutoff;
    }

    this.getShortenedContent = function(){
        return this.body.substring(0, this.getCutoff());
    }

    this.hasMoreContent = function(){
        return this.body.length > this.getShortenedContent().length;
    }
}

function CodeTag(open, close){
    this.open = open;
    this.close = close;
}

function HTMLTag(open, close, length){
    this.open = open;
    this.close = close;
    this.length = length;
}

function getQuestion(title, body, tags, link){
    tagsString = new Array();
    for(i = 0; i < tags.length; i++){
        tagsString.push(tags[i].toString());
    }
    questionsList.push(new Question(title, body, tagsString, link));;
    questionsList[numQuestions].findCodeTags();
    questionsList[numQuestions].findHTMLTags();
    numQuestions++;
}

function displayQuestions(){
    if(numQuestions > 0){
        for(var i = 0; i < numQuestions; i++){
            appendNewResultSkeleton(i);
            var questionSection = questionSections[i];
            $(questionSection).find(".searchResultTitle").html(questionsList[i].title);

            var questionBody = $(questionSection).find(".questionBody");
            $(questionBody).html(questionsList[i].getShortenedContent());

            if(questionsList[i].hasMoreContent()){
                var excerptController = $("<div>").addClass("excerptController").html("More");
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
        var message = $("<h2>").html("Sorry, querying \"" + $('#searchBox').val() + "\" yielded no results. :(");
        $('#questions').append(message);
    }

    uiSettings.updateUI();
}

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

function generateListeners(){
    for(var i = 0; i < questionSections.length; i++){
        $(questionSections[i]).on("click", ".excerptController", function(e){
            var clickedSection = $(this).closest('.searchResultItem');
            var index = $(clickedSection).find('#questionIndex').html();
            var questionBody = $(clickedSection).find('.questionBody');
            if($(this).html() == 'More'){
                $(questionBody).html(questionsList[index].body);
                $(this).html("Less");
                logResultEvent("expand", index);
            }
            else {
                $(questionBody).html(questionsList[index].getShortenedContent());
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

        $(questionSections[i]).on("click", ".questionTags li", function(e){
            searchTags.add($(this).html());
            search(true);
        });

        $(questionSections[i]).on("click", ".searchResultTitle", function(e){
            var clickedSection = $(this).closest('.searchResultItem');
            var index = $(clickedSection).find('#questionIndex').html();
            JavaBridge.openInBrowser(questionsList[index].link);
            logResultEvent("browser", index);
        });
    }

    $(document).on("click", "a", function(e){
        e.preventDefault();
        JavaBridge.openInBrowser($(this).attr('href'));
    });
}

function RegexMatch(result, index, length){
    this.result = result;
    this.index = index;
    this.length = length;
}

String.prototype.regexIndexOf = function(regex, fromIndex){
  var str = fromIndex ? this.substring(fromIndex) : this;
  var match = str.match(regex);

  return new RegexMatch(match ? match[0] : "", match ? str.indexOf(match[0]) + fromIndex : -1, match ? match[0].length : -1);
}