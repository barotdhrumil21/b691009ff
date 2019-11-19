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
	console.log(filename)

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
