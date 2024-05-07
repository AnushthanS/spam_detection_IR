import { useState } from "react";
import axios from 'axios';

let spamProbability = null;
let hamProbability = null;
let indices = null;
let cosineSimilarities = null;

const MainForm = () => {
    const [enteredEmail, setEnteredEmail] = useState("");
    const [resultShow, setResultShow] = useState(false);
    const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

    const submitHandler = event => {
        event.preventDefault();
        setResultShow(false);
        axios.post('http://localhost:8000/predict', { text: enteredEmail })
            .then(response => {
                spamProbability = response.data.spam_percentage;
                hamProbability = response.data.ham_percentage;
                indices = response.data.indices;
                cosineSimilarities = response.data.cosine_similarities;
                setResultShow(true);
                setFeedbackSubmitted(false);
            })
            .catch(error => {
                console.error('There was an error!', error);
            });
    }

    const submitFeedback = (relevant) => {
        axios.post('http://localhost:8000/feedback', { relevant, indices, cosine_similarities: cosineSimilarities })
            .then(() => {
                console.log('Feedback submitted successfully');
                setFeedbackSubmitted(true);
            })
            .catch(error => {
                console.error('There was an error!', error);
            });
    }

    return (
        <>
            <h1 className="text-center text-6xl my-6 mx-auto">Spam Filtering</h1>

            <div className="border-2 border-gray-600 p-6 max-w-lg mx-auto my-6 min-w-fit rounded-lg flex flex-col items-center">
                
                <form onSubmit={submitHandler} className="flex flex-col">
                    <label htmlFor={"inputEmail"} className="font-thin text-2xl">Enter the text to be classified (spam or ham)</label><br /><br />
                    <textarea id={"inputEmail"} value={enteredEmail}
                        onChange={event => setEnteredEmail(event.target.value)} placeholder={"Text Content"} className="border-2 border-gray-100 rounded-xl p-2" /><br /><br />
                    <input type={"submit"} value={"Submit"} className="border-2 p-2 rounded-xl hover:cursor-pointer shadow-md" />
                </form>
            </div>

            {resultShow && <>
                <div className="border-2 border-gray-600 p-6 max-w-lg mx-auto my-6 min-w-fit rounded-lg flex flex-col items-center">
                    <h3 className="font-thin text-2xl">Result:</h3>
                    <p className="font-thin text-xl">{`The text is ${spamProbability}% spam and ${hamProbability}% ham. Hence, the text is likely to be a ${spamProbability > hamProbability ? "SPAM." : "HAM."}`}</p>
                    <button onClick={() => submitFeedback(true)} className="border-2 p-2 rounded-xl mt-4">Agree</button>
                    <button onClick={() => submitFeedback(false)} className="border-2 p-2 rounded-xl mt-4">Disagree</button>
                    {feedbackSubmitted && <p className="font-thin text-xl mt-4">Feedback submitted</p>}
                </div>
            </>

            }


        </>
    );
}

export default MainForm;
