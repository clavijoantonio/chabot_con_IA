const { createBot, createProvider, createFlow, addKeyword } = require('@bot-whatsapp/bot')

const QRPortalWeb = require('@bot-whatsapp/portal')
const BaileysProvider = require('@bot-whatsapp/provider/baileys')
const MockAdapter = require('@bot-whatsapp/database/mock')
const Mysqladater = require('./mysql2.js')
const apiService = require('./src/services/producto.service.js')
const service = new apiService();
const consultaService = require('./src/services/consulta.service.js')
const consulta = new consultaService();
 
const io = require('socket.io-client');
const socket = io('http://localhost:5000');

// Variable para almacenar el historial de conversación
let conversationContext = {
  history: [],
  lastInteraction: null,
  userData: {}
};

// Función principal que maneja toda la conversación
const chatHandler = addKeyword([])
  .addAction(
    { capture: true },
    async (ctx, { flowDynamic, state }) => {
      try {
        // Actualizar contexto con el nuevo mensaje
        conversationContext.history.push({
          user: ctx.body,
          timestamp: new Date().toISOString()
        });
        conversationContext.lastInteraction = 'user';

        // Enviar mensaje al servidor Python con el contexto
        const payload = {
          text: ctx.body,
          context: conversationContext
        };
        
        socket.emit('message', payload);
        console.log('Mensaje enviado con contexto:', payload);

        // Esperar respuesta del servidor Python
        const respuesta = await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => {
            socket.off('response');
            reject(new Error('Tiempo de espera agotado'));
          }, 20000);

          socket.once('response', (data) => {
            clearTimeout(timeout);
            
            // Actualizar contexto con la respuesta
            conversationContext.history.push({
              bot: data.text,
              timestamp: new Date().toISOString()
            });
            conversationContext.lastInteraction = 'bot';
            
            // Guardar cualquier dato adicional del contexto
            if (data.context) {
              conversationContext = {
                ...conversationContext,
                ...data.context
              };
            }

            resolve(data.text);
          });
        });

        // Enviar respuesta al usuario
        await flowDynamic(respuesta);

        // Continuar escuchando sin reiniciar el flujo
        return null;

      } catch (error) {
        console.error('Error en la conversación:', error);
        
        // Manejo elegante del error
        const errorMessage = 'Disculpa, estoy teniendo dificultades. ¿Podrías repetir tu último mensaje?';
        conversationContext.history.push({
          bot: errorMessage,
          timestamp: new Date().toISOString(),
          error: true
        });
        
        await flowDynamic(errorMessage);
        return null;
      }
    }
  );

// Inicialización del chatbot con saludo inicial
const flowPrincipal = addKeyword(['hola', 'hi', 'buenos días', 'buenas tardes'])
  .addAction(async (_, { flowDynamic }) => {
    conversationContext = {
      history: [],
      lastInteraction: null,
      userData: {},
      startedAt: new Date().toISOString()
    };
    
    await flowDynamic('🙌 ¡Hola! Bienvenido/a. ¿En qué puedo ayudarte hoy?');
  })
  .addAction(
    async (ctx, { gotoFlow }) => {
      // Retornar explícitamente la función gotoFlow
      return gotoFlow(chatHandler);
    }
  );
     
      const main = async () => {
        const adapterDB = new MockAdapter()
        //const adapterDB = new Mysqladater()
        const adapterFlow = createFlow([flowPrincipal, chatHandler])
        const adapterProvider = createProvider(BaileysProvider)
        
        createBot({
            flow: adapterFlow,
            provider: adapterProvider,
            database: adapterDB,
        })
        
       
        QRPortalWeb()
    }
    
    main()
    
    
    
    


 




  



    




