var input1 = document.getElementById( 'file-upload' );
var infoArea = document.getElementById( 'inputGroupFile04' );

var input2 = document.getElementById( 'file-upload2' );
var infoArea2 = document.getElementById( 'inputGroupFile05' );

input1.addEventListener( 'change', showFileName );
input2.addEventListener( 'change', showFileName2 );

function showFileName( event ) {
	// the change event gives us the input it occurred in
	var input1 = event.srcElement;
	// the input has an array of files in the `files` property, each one has a name that you can use. We're just using the name here.
	var fileName = input1.files[0].name;


	// use fileName however fits your app best, i.e. add it into a div
	infoArea.textContent  = fileName;

}
function showFileName2( event ) {
	// the change event gives us the input it occurred in
	var input2 = event.srcElement;
	// the input has an array of files in the `files` property, each one has a name that you can use. We're just using the name here.
	var fileName2 = input2.files[0].name;
	// use fileName however fits your app best, i.e. add it into a div
	infoArea2.textContent  = fileName2;
}

var form = document.getElementById('answers');
var fileSelect = document.getElementById('file_select');
var uploadButton = document.getElementById('upload-button');

function fileUpload(form, action_url, div_id) {
    // Create the iframe...
    var iframe = document.createElement("iframe");
    iframe.setAttribute("id", "upload_iframe");
    iframe.setAttribute("name", "upload_iframe");
    iframe.setAttribute("width", "0");
    iframe.setAttribute("height", "0");
    iframe.setAttribute("border", "0");
    iframe.setAttribute("style", "width: 0; height: 0; border: none;");

    // Add to document...
    form.parentNode.appendChild(iframe);
    window.frames['upload_iframe'].name = "upload_iframe";

    iframeId = document.getElementById("upload_iframe");

    // Add event...
    var eventHandler = function () {

            if (iframeId.detachEvent) iframeId.detachEvent("onload", eventHandler);
            else iframeId.removeEventListener("load", eventHandler, false);

            // Message from server...
            if (iframeId.contentDocument) {
                content = iframeId.contentDocument.body.innerHTML;
            } else if (iframeId.contentWindow) {
                content = iframeId.contentWindow.document.body.innerHTML;
            } else if (iframeId.document) {
                content = iframeId.document.body.innerHTML;
            }

            // document.getElementById(div_id).innerHTML = content;

            // Del the iframe...
            setTimeout('iframeId.parentNode.removeChild(iframeId)', 250);
        }

    if (iframeId.addEventListener) iframeId.addEventListener("load", eventHandler, true);
    if (iframeId.attachEvent) iframeId.attachEvent("onload", eventHandler);

    // Set properties of form...
    form.setAttribute("target", "upload_iframe");
    form.setAttribute("action", action_url);
    form.setAttribute("method", "post");
    form.setAttribute("enctype", "multipart/form-data");
    form.setAttribute("encoding", "multipart/form-data");

    // Submit the form...
    form.submit();

    // document.getElementById(div_id).innerHTML = "Uploading...";
}
// $("form#answers").submit(function(e) {
//     e.preventDefault();
// 		var formData = new FormData(this);
// 		console.log($('input[name=csrfmiddlewaretoken]').val());
//     $.ajax({
//         url:"/functions/add-exam-answers/" ,
//         type: 'FILES',
// 				enctype: 'multipart/form-data',
//         data: {csrfmiddlewaretoken: $('input[name=csrfmiddlewaretoken]').val(),
// 								files:fileSelect
// 							},
//
//         success: function (data="formData") {
//             alert(data)
//         },
//     });
// });
